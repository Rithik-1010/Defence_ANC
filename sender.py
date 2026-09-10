import asyncio
import websockets
import numpy as np
import argparse
import time

async def audio_sender(uri, filepath=None, chunk_size=512, sr=16000):
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        if filepath:
            import librosa
            print(f"Streaming from file: {filepath}")
            y, _ = librosa.load(filepath, sr=sr, mono=True)
            
            # Send chunks to simulate real-time
            chunk_duration = chunk_size / sr
            
            for i in range(0, len(y), chunk_size):
                chunk = y[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                start_time = time.time()
                await websocket.send(chunk.tobytes())
                
                # Sleep to maintain real-time pacing
                elapsed = time.time() - start_time
                sleep_time = max(0, chunk_duration - elapsed)
                await asyncio.sleep(sleep_time)
                
            print("Finished sending file.")
        else:
            print("Streaming from live microphone not yet implemented.")
            # Would use sounddevice here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANC Sender")
    parser.add_argument("--uri", default="ws://localhost:8765", help="WebSocket URI")
    parser.add_argument("--file", help="Path to WAV file to stream")
    args = parser.parse_args()
    
    asyncio.run(audio_sender(args.uri, args.file))
