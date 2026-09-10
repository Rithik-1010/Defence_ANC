# AI/ML-Enabled Real-Time Adaptive Noise Cancellation for Mission-Critical Communication
### Technical Plan — Architecture, Dataset, Model, Tools, and Implementation Roadmap

---

## 1. Executive Overview

This document consolidates the full technical plan for the team's Smart India Hackathon submission: a hybrid AI/ML-driven Adaptive Noise Cancellation (ANC) system for defence communication, built on top of the official problem statement's mandated architecture (dual-mic input, complex-domain phase-preserving processing, full-band + sub-band model, reverberation-augmented training, optional LMS/NLMS residual filtering, ONNX/TensorRT edge deployment, and hard targets of SNR>15dB, STOI>0.85, PESQ>2.5), with the team's transient/impulse detector branch and situational-awareness toggle layered on top as differentiators.

The current demo phase runs entirely on two laptops connected over WiFi, standing in for a field unit and a receiving unit, with hardware dumping onto Jetson-class edge devices deferred until after selection.

---

## 2. System Architecture

### 2.1 High-level signal flow

The pipeline is organized as five decoupled stages so each can be built, tested, and demoed independently:

**Capture** — Primary mic (speaker's voice) and reference mic (ambient noise) sampled at 16kHz. On Laptop A, this is either two physical mic inputs, or a software-simulated reference channel built by mixing a pre-recorded defence-noise clip into the primary signal at a controllable SNR.

**Transport** — Audio chunks (sized to match the STFT frame, e.g. 20–32ms) streamed from Laptop A to Laptop B over a WebSocket connection, simulating the tactical wireless link between field and command.

**Pre-processing** — STFT to convert each frame into a complex spectrogram (magnitude + phase, or real + imaginary parts depending on model). Optional MVDR/GSC beamforming stage using both mic channels, documented as designed-in but bypassed in the laptop demo if not yet stabilized.

**Core enhancement** — Three components running together:
- The **backbone neural model** (see Section 3) predicts a complex-valued mask or directly estimates clean complex spectral coefficients, preserving phase throughout.
- The **transient-detector CNN branch** runs in parallel on the same input frames, classifying each frame as impulsive (gunfire, artillery) or continuous (engine, rotor, wind), and switches the suppression strategy — fast attack/release for impulsive, smoother gain-based suppression for continuous.
- The **situational-awareness toggle** is a runtime switch that blends the enhanced signal with a controlled amount of raw ambient signal, so full suppression and partial pass-through are both available without retraining.

**Post-processing and reconstruction** — Optional LMS/NLMS adaptive filter using the reference mic signal to mop up narrowband residuals the neural model misses, followed by inverse STFT with overlap-add to reconstruct the time-domain waveform, which is played out on Laptop B.

### 2.2 Two-laptop demo topology

Laptop A (field unit): mic capture → noise injection (for repeatable, realistic demo conditions) → WebSocket sender.
Laptop B (command/listener unit): WebSocket receiver → full enhancement pipeline above → speaker output → dashboard rendering both raw and enhanced signals, live metrics, and the transient/situational-awareness indicators.

This topology mirrors the real deployment (field radio → enhancement unit at the receiving end) more closely than a single-machine demo, and gives you an honest way to talk about network latency separately from model latency.

### 2.3 Deployment path beyond the demo

Laptop CPU (now) → ONNX export → Jetson Nano/AGX Orin with TensorRT INT8 optimization and I2S dual-mic hardware input (if selected to proceed). The same ONNX graph and Python-side pre/post-processing code carries over unchanged; only the runtime and mic interface change. This is the key point to make to judges — the algorithm and its edge-readiness are already proven independent of which physical box runs it.

---

## 3. Model Selection

### 3.1 Candidates considered

**DTLN** — Two-stage LSTM model, first stage masks in the magnitude STFT domain, second stage refines in a learned time-domain representation. Extremely lightweight and easy to get running fast, well-documented (breizhn/DTLN repo), but doesn't natively operate in the complex spectral domain the way the PS specifies — phase handling is implicit rather than explicit.

**DCCRN** — Deep Complex Convolutional Recurrent Network. Processes real and imaginary spectral components jointly through complex-valued convolutional and recurrent layers, explicitly satisfying the PS's complex-domain phase-preserving requirement. Well-documented, ranked competitively in the DNS Challenge real-time track, moderate compute cost.

**GTCRN** — A more recent, ultra-lightweight complex-domain model explicitly designed for low-power/edge speech enhancement, using grouped/sub-band processing internally. This aligns closely with both the PS's "full-band + sub-band" architecture requirement and the edge-deployment constraint, at lower compute cost than DCCRN.

**FullSubNet-style architectures** — Explicitly split into a full-band branch (captures global spectral envelope/coarse patterns) and a sub-band branch (captures fine-grained per-frequency temporal patterns), which is the most literal match to the PS wording. Heavier to train from scratch within a hackathon timeline.

### 3.2 Recommendation

Use **GTCRN as the primary backbone**. It gives you complex-domain processing and a full-band/sub-band-style internal structure in one lightweight package that's realistic to train and run on Colab/Kaggle GPUs and on laptop CPU for the demo, and it maps naturally onto Jetson deployment later without needing further compression work. Keep **DCCRN as your documented fallback** — mention both in your submission as "primary model GTCRN, with DCCRN evaluated as an alternative," since DCCRN has more tutorials and pretrained checkpoints available if GTCRN training proves unstable under time pressure.

Layer the transient-detector CNN and situational-awareness toggle on top of whichever backbone you finalize — both are architecturally independent additions (a small parallel classifier branch and a runtime mixing switch), so they don't lock you into one backbone choice.

Use classical **MVDR/GSC beamforming** as an optional pre-stage ahead of the neural backbone when both mic channels are genuinely available; skip it gracefully (pass primary-mic-only audio straight to the backbone) when only one channel is usable, and say so explicitly in your Q&A prep.

---

## 4. Dataset Plan

### 4.1 Sources

Clean speech: **LibriSpeech** and **VCTK** (16kHz, large clean corpora, standard for speech enhancement benchmarking).
Defence-adjacent noise: **FSD50K**, **UrbanSound8K**, **ESC-50** for impulsive (gunshots, explosions) and continuous (engines, machinery, wind) classes; supplement with any publicly available military-audio-style datasets you can access for stronger domain realism.
Reverberation: **OpenSLR RIR corpus** to simulate vehicle cabin, cockpit, and outdoor reverberant conditions.

### 4.2 Synthesis pipeline (handles the non-sequential nature of the source data)

Since the speech and noise clips are independent files with no natural pairing, build the training set through **on-the-fly random pairing** rather than any pre-fixed pairing:

Each training sample is constructed by (1) drawing a random speech chunk, (2) drawing a random noise chunk independently, (3) mixing them at a randomly sampled SNR within your target range (e.g. -5dB to 15dB), and (4) optionally convolving with a randomly sampled RIR. This is done inside the PyTorch `Dataset.__getitem__`, so no two epochs see the same pairing, and the source datasets never need to be sequential or pre-aligned.

Chunk all source clips into fixed-length frames (e.g. 4–8 seconds), padding short clips and splitting long ones, before this pairing step, so every sample fed to the model has consistent shape for batching.

Label each synthetic frame with its noise category (impulsive vs continuous) at generation time — this is free supervision for the transient-detector branch, since you already know which noise class you drew.

### 4.3 Splits and validation strategy

Hold out entire speaker and noise-clip identities (not just random frames) for validation and test sets, so the model is evaluated on genuinely unseen speakers and noise instances rather than unseen mixtures of already-seen material. Keep a small "stress test" split with SNRs and noise types deliberately outside the training range, to rehearse how the system behaves on the out-of-distribution audio judges might play live.

---

## 5. Training Plan

### 5.1 Loss function

Use a composite loss combining **SI-SNR** (scale-invariant signal-to-noise ratio, drives overall enhancement quality) with an **STOI-based perceptual loss term** (drives intelligibility specifically), rather than plain MSE, since MSE alone tends to over-smooth speech and hurts perceptual quality even when the numeric loss looks good.

### 5.2 Training infrastructure and schedule

Train on **Google Colab or Kaggle free-tier GPUs**. Given time constraints, plan for short, frequent training runs rather than one long run — checkpoint often, since free-tier sessions can disconnect. Start with a smaller subset of the synthetic data to get the pipeline fully working end-to-end (data loading → model → loss → checkpoint → evaluation) before scaling up data volume; a working small-scale pipeline the night before review is worth more than an unfinished large-scale one.

### 5.3 Evaluation

Compute **SNR, STOI, and PESQ** against the known clean reference signal for every validation sample (this is only possible because your synthetic mixtures retain ground truth — flag clearly in your pitch that live/unseen audio has no such reference, so these metrics describe your validated performance, not a live guarantee). Also track **RTF (real-time factor)**, which must stay below 1.0, and raw per-frame inference latency on both Colab GPU and laptop CPU, since the CPU number is what you'll actually demo.

### 5.4 Practical training tips given the timeline

Get a minimal version training end-to-end first (even with a tiny model and tiny dataset) to catch pipeline bugs early. Use a pretrained/reference implementation of your chosen backbone (official GTCRN or DCCRN repo) as your starting point rather than writing the architecture from scratch — you have a fixed deadline, and adapting a known-working implementation to your data pipeline is a far better use of time than debugging a fresh implementation.

---

## 6. Tools and Technology Stack

Core ML: **PyTorch** for model definition and training, exported via **ONNX** for inference, run through **ONNX Runtime** on both laptop CPU and (later) Jetson.

Audio processing: **librosa** / **soundfile** for file I/O, **numpy/scipy** for STFT/iSTFT, **sounddevice** for live mic capture and playback (simpler streaming API than pyaudio).

Metrics: **pesq** and **pystoi** pip packages for evaluation during training and for live dashboard readouts.

Networking: **websockets** (Python) for streaming audio chunks between the two laptops — reliable on typical WiFi and fast to implement, preferable to raw UDP sockets for a time-constrained build.

Dashboard/UI: **Gradio** for the fastest path to a working interactive demo (built-in audio widgets, before/after players), or **Streamlit** if you want more layout control across multiple tabs (Live Demo / Metrics / Architecture / Hardware Roadmap).

Edge optimization path (post-selection): **NVIDIA TensorRT** for INT8 quantization and layer fusion on Jetson hardware.

---

## 7. Implementation Plan / Timeline

**Phase 1 — Pipeline skeleton (do first):** Get STFT → dummy model (even an identity pass-through) → iSTFT working end-to-end on a single laptop with a pre-recorded file. This validates your plumbing before any real model is involved.

**Phase 2 — Model integration:** Swap in the trained GTCRN/DCCRN backbone via ONNX Runtime once training produces a usable checkpoint; verify output audio is intelligible on real test files before touching the dashboard.

**Phase 3 — Transient detector and toggle:** Add the parallel classifier branch and situational-awareness mixing switch; test both independently against known impulsive and continuous noise clips.

**Phase 4 — Two-laptop networking:** Wrap the working single-machine pipeline with the WebSocket sender/receiver; test on the same WiFi network you'll actually demo on, not just localhost.

**Phase 5 — Dashboard:** Build the Gradio/Streamlit UI last, once the underlying pipeline is proven — the UI should visualize a pipeline that already works, not be built in parallel with a pipeline that doesn't yet.

**Phase 6 — Rehearsal:** Run the full two-laptop demo multiple times end-to-end, including at least one deliberately out-of-distribution noise clip, so you know what the system does when it's stressed rather than finding out live in front of judges.

---

## 8. Risks and Honest Framing for Q&A

**Hardware gap** — Demo runs on laptop, not Jetson; framed as a functional proxy, with ONNX export as the concrete evidence of edge-readiness.

**Live/unseen audio** — PESQ/STOI need a clean reference unavailable in live conditions; the system will still produce enhanced audio, but you can only report numeric quality scores for your validated synthetic test set, and should say this proactively rather than let it be discovered.

**Synthetic data domain gap** — Mixing clean speech with generic noise datasets (rather than genuine field recordings) has known generalization limits; acknowledge this and frame the reverberation augmentation and out-of-distribution stress-testing as your mitigation, not a claim of full realism.

**Network dependency in the demo** — The two-laptop WiFi link is a demo convenience to mirror the field/command split, not a claim that the final system requires networking; the enhancement pipeline itself runs fully on-device regardless of transport.

---

## 9. Submission / Pitch Narrative (usable directly in your write-up)

The system is a hybrid neural-adaptive noise cancellation pipeline for tactical voice communication, combining a lightweight complex-domain neural backbone (GTCRN, with DCCRN evaluated as an alternative) with a parallel transient-detection branch that distinguishes impulsive threats like gunfire from continuous noise like engine or rotor hum, applying a different suppression strategy to each. A situational-awareness toggle lets operators choose between full noise suppression and partial ambient pass-through, reflecting the operational reality that personnel sometimes need to hear their environment. The pipeline is built around the official problem statement's dual-microphone, complex-domain, full-band/sub-band architecture, trained on synthetic mixtures of clean speech and defence-adjacent noise with reverberation augmentation, and validated against SNR, STOI, and PESQ targets. The current prototype runs entirely in software across two laptops connected over WiFi — one simulating the field unit, one the receiving/command unit — as a transparent stand-in for the target Jetson edge hardware; the model is already exported to ONNX, which is the same format used for the eventual TensorRT-optimized Jetson deployment, so the algorithmic work demonstrated here carries over directly to the hardware stage if selected to proceed.
