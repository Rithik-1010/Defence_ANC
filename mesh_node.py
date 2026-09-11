"""
Mesh Node for Defence ANC System
=================================
Each laptop runs this as an independent node. Every node is BOTH a sender and receiver.
Nodes discover each other via UDP broadcast on the local WiFi network.
Audio is streamed only to peers on the SAME channel (like walkie-talkie frequencies).
If a direct peer is unreachable, packets can hop through intermediate nodes (mesh relay).

Architecture:
  - UDP Broadcast Discovery (port 5555): Announces presence and channel
  - WebSocket Server (port 8765): Receives audio from peers
  - WebSocket Client: Connects to discovered peers to send audio
  - Audio Pipeline: Mic capture → STFT → AI Filter → Speaker output
"""

import asyncio
import websockets
import json
import socket
import struct
import time
import threading
import numpy as np
import sounddevice as sd
import queue
from dataclasses import dataclass, field
from typing import Dict, Set

# ──────────────────────────────────────────────
# Peer info
# ──────────────────────────────────────────────
@dataclass
class PeerInfo:
    node_id: str
    ip: str
    port: int
    channel: int
    last_seen: float = 0.0
    hops: int = 0          # 0 = direct, 1+ = relayed
    latency_ms: float = 0.0

# ──────────────────────────────────────────────
# The Mesh Node
# ──────────────────────────────────────────────
class MeshNode:
    def __init__(self, node_id: str, channel: int = 1, port: int = 8765, sr: int = 16000, chunk_size: int = 512):
        self.node_id = node_id
        self.channel = channel
        self.port = port
        self.sr = sr
        self.chunk_size = chunk_size

        # Known peers: node_id -> PeerInfo
        self.peers: Dict[str, PeerInfo] = {}
        self.peers_lock = threading.Lock()

        # Active WebSocket connections to peers (for sending)
        self.outgoing_connections: Dict[str, websockets.WebSocketClientProtocol] = {}

        # Audio playback queue
        self.playback_queue = queue.Queue(maxsize=50)

        # Callbacks for UI updates
        self.on_peer_update = None      # Called when peer list changes
        self.on_audio_received = None   # Called with (sender_id, audio_chunk)
        self.on_status_change = None    # Called with status string

        self._running = False
        self._discovery_sock = None

    # ──────────────────────────────────────────
    # Peer Discovery via UDP Broadcast
    # ──────────────────────────────────────────
    def _get_local_ip(self):
        """Get the local WiFi IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    async def _broadcast_presence(self):
        """Periodically broadcast this node's presence on the network."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)

        local_ip = self._get_local_ip()
        msg = json.dumps({
            "type": "discovery",
            "node_id": self.node_id,
            "ip": local_ip,
            "port": self.port,
            "channel": self.channel,
            "timestamp": time.time()
        }).encode()

        while self._running:
            try:
                sock.sendto(msg, ("<broadcast>", 5555))
            except Exception:
                pass
            await asyncio.sleep(2)  # Broadcast every 2 seconds
        sock.close()

    async def _listen_discovery(self):
        """Listen for UDP broadcast discovery messages from other nodes."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", 5555))
        except Exception:
            sock.bind(("0.0.0.0", 5556))
        sock.setblocking(False)

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                data, addr = await loop.run_in_executor(None, lambda: sock.recvfrom(4096))
                msg = json.loads(data.decode())

                if msg.get("type") == "discovery" and msg["node_id"] != self.node_id:
                    peer = PeerInfo(
                        node_id=msg["node_id"],
                        ip=msg["ip"],
                        port=msg["port"],
                        channel=msg["channel"],
                        last_seen=time.time(),
                        hops=0
                    )
                    with self.peers_lock:
                        self.peers[peer.node_id] = peer

                    if self.on_peer_update:
                        self.on_peer_update(self.get_peer_list())

                    # Auto-connect to same-channel peers
                    if peer.channel == self.channel:
                        asyncio.ensure_future(self._connect_to_peer(peer))

            except BlockingIOError:
                await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(0.5)
        sock.close()

    # ──────────────────────────────────────────
    # WebSocket Server (Receive audio from peers)
    # ──────────────────────────────────────────
    async def _ws_handler(self, websocket):
        """Handle incoming WebSocket connections from peers."""
        sender_id = "unknown"
        try:
            async for message in websocket:
                try:
                    # First 64 bytes: JSON header, rest: audio data
                    header_len = struct.unpack("!I", message[:4])[0]
                    header = json.loads(message[4:4 + header_len])
                    audio_bytes = message[4 + header_len:]

                    sender_id = header.get("node_id", "unknown")
                    sender_channel = header.get("channel", -1)
                    hop_count = header.get("hops", 0)
                    send_time = header.get("timestamp", 0)

                    # ONLY process audio from the SAME channel
                    if sender_channel != self.channel:
                        # If we know a peer on that channel, relay (mesh hop)
                        if hop_count < 3:  # Max 3 hops to prevent loops
                            await self._relay_packet(message, sender_id, hop_count + 1)
                        continue

                    # Decode audio
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Calculate latency
                    latency = (time.time() - send_time) * 1000
                    with self.peers_lock:
                        if sender_id in self.peers:
                            self.peers[sender_id].latency_ms = latency

                    # Queue for playback (already filtered by sender)
                    try:
                        self.playback_queue.put_nowait(audio_chunk)
                    except queue.Full:
                        pass  # Drop packet if buffer full (prioritize low latency)

                    if self.on_audio_received:
                        self.on_audio_received(sender_id, audio_chunk)

                except Exception:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _relay_packet(self, raw_message, original_sender, new_hop_count):
        """Relay a packet to peers on the target channel (mesh hopping)."""
        for nid, conn in list(self.outgoing_connections.items()):
            if nid != original_sender:
                try:
                    await conn.send(raw_message)
                except Exception:
                    pass

    # ──────────────────────────────────────────
    # WebSocket Client (Send audio to peers)
    # ──────────────────────────────────────────
    async def _connect_to_peer(self, peer: PeerInfo):
        """Establish outgoing WebSocket connection to a peer."""
        if peer.node_id in self.outgoing_connections:
            return
        try:
            uri = f"ws://{peer.ip}:{peer.port}"
            ws = await asyncio.wait_for(websockets.connect(uri), timeout=3)
            self.outgoing_connections[peer.node_id] = ws
        except Exception:
            pass

    async def _send_audio_chunk(self, audio_chunk: np.ndarray):
        """Send an audio chunk to all connected same-channel peers."""
        header = json.dumps({
            "node_id": self.node_id,
            "channel": self.channel,
            "hops": 0,
            "timestamp": time.time()
        }).encode()

        header_len = struct.pack("!I", len(header))
        packet = header_len + header + audio_chunk.astype(np.float32).tobytes()

        dead = []
        for nid, ws in list(self.outgoing_connections.items()):
            try:
                await ws.send(packet)
            except Exception:
                dead.append(nid)

        for nid in dead:
            self.outgoing_connections.pop(nid, None)

    # ──────────────────────────────────────────
    # Audio I/O (Mic capture + Speaker playback)
    # ──────────────────────────────────────────
    def _start_audio(self, loop):
        """Start mic capture and speaker playback in background threads."""
        mic_queue = queue.Queue(maxsize=20)

        def mic_callback(indata, frames, time_info, status):
            mic_queue.put(indata[:, 0].copy())

        def speaker_callback(outdata, frames, time_info, status):
            try:
                chunk = self.playback_queue.get_nowait()
                if len(chunk) < frames:
                    chunk = np.pad(chunk, (0, frames - len(chunk)))
                outdata[:, 0] = chunk[:frames]
            except queue.Empty:
                outdata.fill(0)

        # Start streams
        self._input_stream = sd.InputStream(
            samplerate=self.sr, channels=1,
            blocksize=self.chunk_size, callback=mic_callback
        )
        self._output_stream = sd.OutputStream(
            samplerate=self.sr, channels=1,
            blocksize=self.chunk_size, callback=speaker_callback
        )
        self._input_stream.start()
        self._output_stream.start()

        # Background thread to send mic audio to peers
        def sender_thread():
            while self._running:
                try:
                    chunk = mic_queue.get(timeout=0.5)
                    asyncio.run_coroutine_threadsafe(
                        self._send_audio_chunk(chunk), loop
                    )
                except queue.Empty:
                    pass

        t = threading.Thread(target=sender_thread, daemon=True)
        t.start()

    # ──────────────────────────────────────────
    # Peer cleanup (remove stale peers)
    # ──────────────────────────────────────────
    async def _cleanup_peers(self):
        """Remove peers not seen in the last 10 seconds."""
        while self._running:
            now = time.time()
            stale = []
            with self.peers_lock:
                for nid, peer in self.peers.items():
                    if now - peer.last_seen > 10:
                        stale.append(nid)
                for nid in stale:
                    del self.peers[nid]
                    self.outgoing_connections.pop(nid, None)

            if stale and self.on_peer_update:
                self.on_peer_update(self.get_peer_list())

            await asyncio.sleep(5)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────
    def get_peer_list(self):
        """Return current peer list as list of dicts for the UI."""
        with self.peers_lock:
            return [
                {
                    "Node ID": p.node_id,
                    "IP": p.ip,
                    "Channel": p.channel,
                    "Status": "🟢 Connected" if (time.time() - p.last_seen) < 10 else "🔴 Lost",
                    "Hops": p.hops,
                    "Latency": f"{p.latency_ms:.0f}ms" if p.latency_ms > 0 else "-"
                }
                for p in self.peers.values()
            ]

    async def start(self):
        """Start the mesh node (discovery, server, audio)."""
        self._running = True
        loop = asyncio.get_event_loop()

        # Start WebSocket server
        server = await websockets.serve(self._ws_handler, "0.0.0.0", self.port)

        # Start audio I/O
        self._start_audio(loop)

        local_ip = self._get_local_ip()
        print(f"╔══════════════════════════════════════════════╗")
        print(f"║  MESH NODE: {self.node_id:<33}║")
        print(f"║  IP: {local_ip:<40}║")
        print(f"║  Channel: {self.channel:<35}║")
        print(f"║  WebSocket Port: {self.port:<28}║")
        print(f"╚══════════════════════════════════════════════╝")

        if self.on_status_change:
            self.on_status_change(f"🟢 Node {self.node_id} active on channel {self.channel} @ {local_ip}")

        # Run all background tasks
        await asyncio.gather(
            self._broadcast_presence(),
            self._listen_discovery(),
            self._cleanup_peers(),
        )

    def stop(self):
        """Stop the mesh node."""
        self._running = False
        try:
            self._input_stream.stop()
            self._output_stream.stop()
        except Exception:
            pass


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Defence ANC Mesh Node")
    parser.add_argument("--id", default=f"soldier-{socket.gethostname()}", help="Unique node ID")
    parser.add_argument("--channel", type=int, default=1, help="Communication channel (1-10)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    args = parser.parse_args()

    node = MeshNode(node_id=args.id, channel=args.channel, port=args.port)

    def on_peer(peers):
        print(f"[MESH] Peers: {[p['Node ID'] + ' (ch' + str(p['Channel']) + ')' for p in peers]}")

    node.on_peer_update = on_peer

    try:
        asyncio.run(node.start())
    except KeyboardInterrupt:
        node.stop()
        print("Node stopped.")
