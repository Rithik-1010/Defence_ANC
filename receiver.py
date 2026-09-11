import asyncio
import websockets
import numpy as np
import sounddevice as sd
import argparse
import torch
import queue
from pipeline import AudioPipeline
from model_gtcrn import ComplexMasker

# Global configuration
SR = 16000
CHUNK_SIZE = 256 # 16ms hop length for a 32ms frame at 16kHz

pipeline = AudioPipeline(target_sr=SR)
model = ComplexMasker()
model.eval()

# Thread-safe queue for the audio playback thread
sync_playback_queue = queue.Queue()

# Overlap-add input buffer
input_buffer = np.zeros(pipeline.frame_size, dtype=np.float32)

def process_chunk(new_samples):
    global input_buffer
    
    # Shift buffer left and append new samples
    input_buffer = np.roll(input_buffer, -CHUNK_SIZE)
    input_buffer[-CHUNK_SIZE:] = new_samples
    
    # 1. STFT
    complex_spec = pipeline.stft_frame(input_buffer)
    
    # 2. AI Model Inference (DCCRN mock)
    # Target shape: (Batch=1, Freq=257, Time=1)
    spec_tensor = torch.tensor(complex_spec, dtype=torch.complex64).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        enhanced_tensor = model(spec_tensor)
    
    enhanced_spec = enhanced_tensor.squeeze().numpy()
    
    # 3. iSTFT and Overlap-Add
    out_samples = pipeline.istft_overlap_add(enhanced_spec)
    
    # WOLA Gain compensation
    wola_gain = np.sum(pipeline.window**2) / (pipeline.frame_size / pipeline.hop_length)
    return out_samples / wola_gain

def playback_callback(outdata, frames, time, status):
    if status:
        print(f"Speaker status: {status}")
    try:
        # Get processed chunk and send to speaker
        chunk = sync_playback_queue.get_nowait()
        outdata[:, 0] = chunk
    except queue.Empty:
        # If queue is empty (network delay or packet loss), play silence
        outdata.fill(0)

async def handler(websocket):
    print("Sender connected! Streaming live audio...")
    
    # Start the live speaker output stream
    with sd.OutputStream(samplerate=SR, channels=1, blocksize=CHUNK_SIZE, callback=playback_callback):
        try:
            async for message in websocket:
                chunk = np.frombuffer(message, dtype=np.float32)
                
                if len(chunk) != CHUNK_SIZE:
                    continue
                
                # Pass through the real-time AI noise cancellation pipeline
                enhanced_chunk = process_chunk(chunk)
                
                # Queue for immediate playback
                sync_playback_queue.put(enhanced_chunk)
                
        except websockets.exceptions.ConnectionClosed:
            print("Sender disconnected.")

async def main(host, port):
    print(f"ANC Receiver listening on ws://{host}:{port}")
    # Note: Using modern websockets signature for serve
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANC Receiver (Live Speaker & AI Filter)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        print("Receiver stopped.")
