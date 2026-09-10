# Agent Build Prompt — Real-Time Adaptive Noise Cancellation System

You are building a working prototype of an AI/ML-based real-time Adaptive Noise Cancellation (ANC) system for defence voice communication. This will be demoed across two laptops connected over WiFi (one simulating a field unit, one a receiving/command unit), running entirely on CPU for now, with all components structured so they later port unchanged to a Jetson edge device via ONNX/TensorRT. Build incrementally and get each phase fully working before starting the next — do not build the dashboard before the pipeline works, and do not add networking before the single-machine pipeline is proven on a file.

## Context and constraints

- Language/stack: Python, PyTorch for training, ONNX + ONNX Runtime for inference, numpy/scipy/librosa/soundfile for audio, sounddevice for live capture/playback, websockets for the inter-laptop link, Gradio (preferred) or Streamlit for the dashboard.
- Target sample rate: 16kHz throughout.
- No internet/hardware dependency assumptions beyond a local WiFi network between two machines and standard laptop mics/speakers.
- Everything must run in real time (or near real time) on ordinary laptop CPUs — no GPU assumption at inference time.
- Keep the codebase modular: capture, streaming, preprocessing, model inference, transient detection, post-processing, metrics, and UI must each be separate, independently testable modules/files, since components will be swapped and demoed in isolation.

## Phase 1 — Core offline pipeline (build and verify first)

Build a script that takes a single noisy WAV file, and:
1. Loads audio, resamples to 16kHz mono if needed.
2. Runs STFT with configurable frame size/hop (start with 32ms frame, 50% overlap).
3. Passes the complex spectrogram through a placeholder pass-through "model" (identity function) so the plumbing can be verified before any real model exists.
4. Reconstructs the waveform via inverse STFT with overlap-add.
5. Saves the output WAV and confirms it's audibly identical to the input (sanity check on the STFT/iSTFT round trip).

Deliverable: a `pipeline.py` (or module) with clearly separated `stft_frame()`, `istft_overlap_add()`, and `process_stream()` functions, plus a small CLI or script to run it end to end on a test file.

## Phase 2 — Dataset synthesis pipeline

Build a PyTorch `Dataset` that does NOT require paired/sequential source files. Given a directory of clean speech clips (e.g. LibriSpeech/VCTK subset) and a directory of categorized noise clips (impulsive: gunfire/explosion-type clips; continuous: engine/rotor/wind-type clips, e.g. from FSD50K/UrbanSound8K/ESC-50), and optionally a directory of RIR files (e.g. OpenSLR RIR corpus):

- Chunk all source clips into fixed-length frames (configurable, default 4 seconds), discarding or padding as needed.
- On each `__getitem__` call, randomly draw one speech chunk and one noise chunk independently (no fixed pairing), randomly sample a target SNR from a configurable range (default -5dB to 15dB), mix them at that SNR, and with some configurable probability convolve the mix with a randomly drawn RIR.
- Return the noisy mixture, the clean reference, the SNR used, and the noise category label (impulsive/continuous) for that sample — the label is known automatically since you know which noise directory the clip came from.
- Support a held-out validation/test split by speaker ID and noise-clip ID (not by random frame), so validation never reuses a speaker or noise clip seen in training.

Deliverable: `dataset.py` with the `Dataset` class, a config for paths and mixing parameters, and a quick script that draws and plays/saves a few sample mixtures for manual sanity-checking.

## Phase 3 — Model: GTCRN backbone (primary), DCCRN as fallback

Implement or adapt an existing open-source implementation of **GTCRN** (a lightweight complex-domain speech enhancement model with grouped/sub-band internal structure) as the primary backbone, operating on the complex STFT representation (real + imaginary or magnitude + phase) and predicting a complex mask or direct clean spectral estimate. If GTCRN proves unstable to train in the available time, fall back to an existing **DCCRN** implementation with the same input/output interface, so the rest of the pipeline doesn't need to change.

- Model interface: takes a batch of noisy complex spectrograms, returns enhanced complex spectrograms (or masks to apply to the input).
- Loss: composite of SI-SNR (on reconstructed time-domain signal) and an STOI-based perceptual term. Implement or use existing differentiable approximations for STOI in the loss if needed.
- Training loop: standard PyTorch loop, checkpointing every N steps (assume Colab/Kaggle sessions can disconnect), logging training/validation SI-SNR, STOI, PESQ periodically.
- Export: after training, export the model to ONNX and verify it loads and runs correctly through ONNX Runtime, producing near-identical output to the PyTorch model on a test batch.

Deliverable: `model_gtcrn.py`, `train.py`, `export_onnx.py`, and a checkpoint/ONNX file once trained.

## Phase 4 — Transient/impulse detector branch

Build a small CNN classifier that takes the same per-frame spectral features as the backbone and outputs a binary (or per-frame probability) classification: impulsive vs continuous noise. Train it on the same synthetic dataset using the noise-category label already available from Phase 2. Its output should be usable to switch the post-processing suppression strategy: fast attack/release when impulsive is detected, smoother suppression otherwise.

Deliverable: `transient_detector.py` with its own small training script, since it can train faster/separately from the main backbone.

## Phase 5 — Situational-awareness toggle and residual filter

Implement a runtime mixing function that blends the enhanced output with a controllable amount of the raw input signal (0% = full suppression, up to some max % = partial ambient pass-through), exposed as a single parameter the UI can control live. Also implement an optional LMS/NLMS adaptive filter stage (using the reference/second channel if available) applied after the neural model, as a residual-noise safety net — make this toggleable too, since it depends on a genuine second mic channel being available.

Deliverable: `postprocess.py` with `apply_awareness_mix()` and `lms_residual_filter()` functions.

## Phase 6 — Networking between two laptops

Build a WebSocket sender (runs on the "field" laptop) that captures live mic audio (or streams a pre-recorded noisy file for demo reliability) in chunks matching the STFT frame size, and a WebSocket receiver (runs on the "command" laptop) that receives chunks, runs them through the full pipeline (model + transient detector + postprocess), and plays the result out through speakers with minimal added buffering delay. Test on localhost first, then on the actual WiFi network intended for the demo. Measure and log end-to-end latency (capture to playback) separately from model inference latency.

Deliverable: `sender.py`, `receiver.py`, and a short README on how to run both together.

## Phase 7 — Dashboard

Build a Gradio (or Streamlit) app, running on the receiving laptop, with:
- A live/before-after audio comparison (raw vs enhanced).
- A live waveform and/or spectrogram view.
- A visible indicator showing the current transient-detector classification (impulsive/continuous) as it updates.
- A situational-awareness slider/toggle wired to the Phase 5 mixing function.
- A metrics panel showing running SNR/STOI/PESQ against the known clean reference (for pre-recorded test material) and current end-to-end latency.
- A static panel describing the hardware roadmap (laptop today → Jetson + TensorRT if selected).

Deliverable: `app.py` runnable with a single command, launching the full demo UI wired to the live pipeline from Phase 6.

## Acceptance criteria for the overall build

- Each phase's script runs standalone and has been verified before the next phase depends on it.
- The two-laptop demo runs reliably on the actual WiFi network to be used, not just localhost.
- Reported SNR/STOI/PESQ numbers are real measured values from the validation set, not placeholders.
- The system has been tested against at least one deliberately out-of-distribution noise clip so its failure behavior is known ahead of time.
- All model code has a clear, documented path from PyTorch checkpoint to ONNX to ONNX Runtime inference, since this is the evidence of edge-readiness for the pitch.

Work phase by phase, confirm each phase is functioning before moving to the next, and flag clearly if any phase (especially model training) needs to be simplified or descoped given time constraints, rather than silently cutting corners.
