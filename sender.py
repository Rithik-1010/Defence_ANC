import asyncio
import websockets
import numpy as np
import sounddevice as sd
import argparse

async def audio_sender(uri, sr=16000, chunk_size=256):
    queue = asyncio.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"Mic status: {status}")
        # Put the mono audio chunk into the queue
        queue.put_nowait(indata[:, 0].copy())

    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}.")
        print(f"Capturing live microphone audio at {sr}Hz...")
        
        # Start the microphone stream
        with sd.InputStream(samplerate=sr, channels=1, blocksize=chunk_size, callback=callback):
            while True:
                chunk = await queue.get()
                # Send the float32 numpy array as bytes
                await websocket.send(chunk.astype(np.float32).tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANC Sender (Live Mic)")
    parser.add_argument("--uri", default="ws://localhost:8765", help="WebSocket URI of the receiver")
    args = parser.parse_args()
    
    try:
        asyncio.run(audio_sender(args.uri))
    except KeyboardInterrupt:
        print("Stopped live stream.")
