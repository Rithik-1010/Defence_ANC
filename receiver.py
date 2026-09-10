import asyncio
import websockets
import numpy as np
import argparse
from pipeline import AudioPipeline

# Global configuration
CHUNK_SIZE = 512
SR = 16000

async def handler(websocket, path):
    print("Client connected!")
    pipeline = AudioPipeline(target_sr=SR)
    
    try:
        async for message in websocket:
            # Reconstruct float32 array from bytes
            chunk = np.frombuffer(message, dtype=np.float32)
            
            # Process chunk through the pipeline (simplified here)
            # In a full implementation, you'd buffer chunks to match frame_size/hop
            # and then run STFT -> Model -> iSTFT
            
            # For now, just a dummy passthrough to demonstrate networking
            # and that we receive the data
            processed_chunk = chunk
            
            # Here you would typically play it back via sounddevice
            # or send it to the dashboard for visualization
            
            # Send back acknowledgment or processed data if needed
            # await websocket.send("ACK")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main(host, port):
    print(f"Starting server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANC Receiver Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()
    
    asyncio.run(main(args.host, args.port))
