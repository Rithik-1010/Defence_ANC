import numpy as np

def apply_awareness_mix(enhanced_signal, raw_signal, mix_ratio):
    """
    Blends the enhanced output with the raw input for situational awareness.
    mix_ratio: 0.0 (full suppression) to 1.0 (full raw signal)
    """
    # Ensure they are the same length
    min_len = min(len(enhanced_signal), len(raw_signal))
    enh = enhanced_signal[:min_len]
    raw = raw_signal[:min_len]
    
    # Simple linear mix
    mixed = (1.0 - mix_ratio) * enh + mix_ratio * raw
    return mixed

def lms_residual_filter(primary_signal, reference_signal, mu=0.01, filter_len=32):
    """
    Optional Normalized Least Mean Squares (NLMS) filter for residual noise.
    Used if a second reference microphone channel is available.
    """
    n = len(primary_signal)
    w = np.zeros(filter_len)
    output = np.zeros(n)
    
    # Pad reference with zeros at the beginning for the filter delay line
    ref_padded = np.pad(reference_signal, (filter_len-1, 0), mode='constant')
    
    for i in range(n):
        x_vec = ref_padded[i:i+filter_len][::-1] # Delay line
        
        # Power of the input vector for normalization
        power = np.dot(x_vec, x_vec) + 1e-6
        
        # Estimate noise
        y = np.dot(w, x_vec)
        
        # Error (this is our enhanced signal)
        e = primary_signal[i] - y
        output[i] = e
        
        # Update weights (NLMS)
        w = w + (mu / power) * e * x_vec
        
    return output

if __name__ == "__main__":
    # Test awareness mix
    raw = np.ones(100)
    enh = np.zeros(100)
    print("Mix 50%:", apply_awareness_mix(enh, raw, 0.5)[:5])
    
    # Test LMS
    ref = np.random.randn(1000)
    # Primary is a delayed version of ref plus some "clean" signal
    clean = np.sin(np.linspace(0, 10, 1000))
    prim = np.roll(ref, 5) * 0.5 + clean
    
    filtered = lms_residual_filter(prim, ref, mu=0.1)
    # The output should have the noise removed and be mostly the clean sine wave
    print("LMS filtered first 5 samples:", filtered[:5])
