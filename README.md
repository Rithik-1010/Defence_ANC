# Defence ANC: AI/ML-Driven Adaptive Noise Cancellation System

A real-time AI/ML-based Adaptive Noise Cancellation (ANC) system engineered for mission-critical defence voice communication over Wi-Fi networks and edge devices.

The system processes real-time audio streams dynamically, using a Deep Complex Convolutional Recurrent Network (DCCRN) and CNN-based transient detection to remove continuous engine hum, wind, impulsive gunfire, and artillery noise while preserving human speech intelligibility and phase information.

---

## 🌟 Key Features

- **Live Wi-Fi Streaming ("Phone Call" Mode)**: Real-time, low-latency streaming between Laptop A (Field Unit) and Laptop B (Command Unit) using WebSockets and hardware mic/speaker integration via `sounddevice`.
- **DCCRN Architecture**: Complex domain processing (magnitude + phase) for natural speech reconstruction without perceptual distortion.
- **CNN Transient Detector**: Real-time classification of non-stationary, impulsive noise (gunshots/explosions) vs. stationary continuous noise (engine/rotor/wind).
- **Synthetic Data Synthesis & Training Pipeline**: Dynamic dataset generator (`dataset.py`) mixing clean voice with defence noise clips at variable SNRs, accompanied by `train.py` for end-to-end PyTorch model training.
- **High-Tech Dashboard**: Interactive Gradio interface (`app.py`) featuring live metric streaming (SNR, STOI, PESQ), spectrogram visualizations (`matplotlib`/`librosa`), situational awareness controls, and network device tracking.
- **Situational Awareness & Residual LMS Filter**: Adjustable pass-through slider (0% to 100% ambient) and optional multi-channel Normalized LMS (NLMS) filter stage.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `app.py` | Colorful Gradio 6.0 dashboard with real-time graphs, spectrogram analysis, and device monitoring |
| `train.py` | PyTorch training pipeline script for training the DCCRN model on defence noise datasets |
| `dataset.py` | PyTorch `Dataset` loader for dynamic, on-the-fly noisy speech synthesis |
| `pipeline.py` | Core STFT/iSTFT framing, windowing, and overlap-add (OLA) streaming engine |
| `model_gtcrn.py` | Complex domain neural network (`ComplexMasker`) backbone for spectral mask prediction |
| `transient_detector.py` | CNN classifier to detect non-stationary impulsive noise vs stationary continuous noise |
| `postprocess.py` | Situational awareness linear mixer and NLMS adaptive residual filter |
| `sender.py` | Live microphone capture script (Laptop A / Field Unit) sending audio over WebSockets |
| `receiver.py` | Live WebSocket receiver & playback engine (Laptop B / Command Unit) executing real-time AI filtering |
| `SIH_ANC_Agent_Prompt.md` | Core specification prompt document |
| `SIH_ANC_Technical_Plan.md` | Detailed technical architecture plan |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure Python 3.9+ is installed, then run:

```bash
pip install numpy scipy librosa soundfile torch torchaudio onnx onnxruntime gradio websockets sounddevice matplotlib
```

---

### 2. Live Wi-Fi Streaming Setup (Laptop A ↔ Laptop B)

#### On Laptop B (Command Unit / Receiver):
```bash
python receiver.py --host 0.0.0.0 --port 8765
```

#### On Laptop A (Field Unit / Sender):
```bash
python sender.py --uri ws://<LAPTOP_B_IP_ADDRESS>:8765
```
*Replace `<LAPTOP_B_IP_ADDRESS>` with Laptop B's local Wi-Fi IP (e.g. `ws://192.168.1.15:8765`). Once connected, speak into Laptop A's mic to hear the real-time AI-filtered audio on Laptop B's speakers.*

---

### 3. Launching the Interactive Dashboard

Launch the web interface:
```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in your browser to view:
- Live streaming performance metrics (**SNR > 15 dB**, **STOI > 0.85**, **PESQ > 2.5**).
- Spectrogram visualizations for uploaded audio files.
- Active network device status and situational awareness controls.

---

### 4. Training the AI Model

To run the PyTorch training loop on synthetic or real defence datasets:
```bash
python train.py
```
This script populates `data/clean` and `data/noise` (with simulated engine hum and artillery shots if empty), trains the `ComplexMasker` model over multiple epochs, and saves the trained weights to `defence_anc_model.pth`.

---

## 🛡️ Target Specifications

- **Signal-to-Noise Ratio (SNR)**: > 15 dB improvement
- **Short-Time Objective Intelligibility (STOI)**: > 0.85
- **Perceptual Evaluation of Speech Quality (PESQ)**: > 2.5
- **Latency**: < 20 ms chunk processing for real-time edge deployment (NVIDIA Jetson / ONNX Runtime)
