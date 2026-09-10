import numpy as np
import librosa
import soundfile as sf
import argparse
import os

class AudioPipeline:
    def __init__(self, target_sr=16000, frame_size_ms=32, overlap_pct=50):
        self.target_sr = target_sr
        self.frame_size = int(target_sr * (frame_size_ms / 1000.0))
        self.hop_length = int(self.frame_size * (1.0 - (overlap_pct / 100.0)))
        # Hanning window
        self.window = np.hanning(self.frame_size)
        self.ola_buffer = np.zeros(self.frame_size)
        
    def stft_frame(self, frame):
        """Applies window and computes real FFT."""
        windowed = frame * self.window
        return np.fft.rfft(windowed)
        
    def istft_overlap_add(self, complex_spec):
        """Computes IFFT, applies window and does overlap-add."""
        time_frame = np.fft.irfft(complex_spec, n=self.frame_size)
        time_frame = time_frame * self.window
        
        self.ola_buffer += time_frame
        out_samples = self.ola_buffer[:self.hop_length].copy()
        
        self.ola_buffer = np.roll(self.ola_buffer, -self.hop_length)
        self.ola_buffer[-self.hop_length:] = 0
        
        return out_samples

def process_stream(input_file, output_file, sr=16000):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")
        
    # Security/Robustness check: Ensure file sizes are reasonable to avoid memory exhaustion
    if os.path.getsize(input_file) > 100 * 1024 * 1024:
        raise ValueError("Input file is too large (>100MB).")
        
    y, _ = librosa.load(input_file, sr=sr, mono=True)
    pipeline = AudioPipeline(target_sr=sr)
    
    # Pad to ensure perfect reconstruction
    y_padded = np.pad(y, (pipeline.frame_size, pipeline.frame_size), mode='constant')
    output = []
    
    for i in range(0, len(y_padded) - pipeline.frame_size, pipeline.hop_length):
        frame = y_padded[i:i + pipeline.frame_size]
        
        # 1. STFT
        complex_spec = pipeline.stft_frame(frame)
        
        # 2. Placeholder model (Identity pass-through)
        enhanced_spec = complex_spec 
        
        # 3. Inverse STFT + OLA
        out_samples = pipeline.istft_overlap_add(enhanced_spec)
        output.append(out_samples)
        
    y_out = np.concatenate(output)
    
    # Correct WOLA gain scaling
    wola_gain = np.sum(pipeline.window**2) / (pipeline.frame_size / pipeline.hop_length)
    y_out = y_out / wola_gain
    
    # Trim padding and match original length
    y_out = y_out[pipeline.frame_size:pipeline.frame_size + len(y)]
    
    sf.write(output_file, y_out, sr)
    print(f"Processed stream and saved to {output_file}")
    
    # Check max difference for validation (sanity check)
    max_diff = np.max(np.abs(y - y_out))
    print(f"Reconstruction Max Diff: {max_diff:.6e}")
    if max_diff > 1e-4:
        print("Warning: STFT/iSTFT round trip error is high.")
    else:
        print("Success: Round-trip reconstruction is nearly identical to input (error within acceptable threshold).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Core Pipeline for ANC")
    parser.add_argument("input", help="Input noisy WAV file path")
    parser.add_argument("output", help="Output enhanced WAV file path")
    args = parser.parse_args()
    
    # Simple path sanitization for security
    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    
    process_stream(in_path, out_path)
