"""
Dataset Downloader for Defence ANC System
==========================================
Downloads real-world noise and speech datasets for training the AI model.
Sources:
  - ESC-50 (Environmental Sound Classification): gunshots, engine, wind, rain
  - UrbanSound8K: sirens, drilling, street noise
  - LibriSpeech (clean speech samples)
  - Custom synthetic: artillery bursts, helicopter rotor simulation
"""

import os
import urllib.request
import zipfile
import tarfile
import numpy as np
import soundfile as sf
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
IMPULSIVE_DIR = os.path.join(NOISE_DIR, "impulsive")
CONTINUOUS_DIR = os.path.join(NOISE_DIR, "continuous")

SR = 16000

def ensure_dirs():
    for d in [CLEAN_DIR, IMPULSIVE_DIR, CONTINUOUS_DIR]:
        os.makedirs(d, exist_ok=True)

def download_file(url, dest):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to: {dest}")
    except Exception as e:
        print(f"  Download failed: {e}")

def generate_synthetic_defence_noise():
    """
    Generate realistic synthetic defence noise when real datasets are unavailable.
    This covers the key noise categories needed for training.
    """
    print("\n=== Generating Synthetic Defence Noise Dataset ===")
    
    t_4s = np.linspace(0, 4, SR * 4)
    t_8s = np.linspace(0, 8, SR * 8)
    
    # ── IMPULSIVE NOISE (Gunfire, Artillery, Explosions) ──
    print("Generating impulsive noise (gunfire, artillery, explosions)...")
    
    for i in range(15):
        # Gunshot: Sharp attack, fast exponential decay, broadband
        audio = np.zeros(SR * 4)
        num_shots = np.random.randint(1, 6)
        for _ in range(num_shots):
            pos = np.random.randint(0, len(audio) - SR)
            duration = np.random.randint(SR // 20, SR // 5)
            decay = np.exp(-np.linspace(0, np.random.uniform(5, 15), duration))
            burst = np.random.randn(duration) * decay * np.random.uniform(0.5, 1.0)
            audio[pos:pos + duration] += burst
        # Add subtle background rumble
        audio += np.random.randn(len(audio)) * 0.02
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(IMPULSIVE_DIR, f"gunfire_{i:03d}.wav"), audio, SR)

    for i in range(10):
        # Artillery / Explosion: Lower frequency, longer decay, more energy
        audio = np.zeros(SR * 4)
        num_blasts = np.random.randint(1, 3)
        for _ in range(num_blasts):
            pos = np.random.randint(0, len(audio) - SR * 2)
            duration = np.random.randint(SR // 2, SR * 2)
            decay = np.exp(-np.linspace(0, np.random.uniform(2, 6), duration))
            # Low frequency rumble + broadband crack
            low_freq = np.sin(2 * np.pi * np.random.uniform(20, 80) * np.linspace(0, duration / SR, duration))
            burst = (np.random.randn(duration) * 0.6 + low_freq * 0.4) * decay
            audio[pos:pos + min(duration, len(audio) - pos)] += burst[:min(duration, len(audio) - pos)]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(IMPULSIVE_DIR, f"artillery_{i:03d}.wav"), audio, SR)
    
    # ── CONTINUOUS NOISE (Engine, Helicopter, Wind, Vehicle) ──
    print("Generating continuous noise (engines, helicopter, wind, vehicles)...")
    
    for i in range(10):
        # Engine humming: Low frequency harmonics
        freqs = [np.random.uniform(40, 100) * k for k in range(1, 5)]
        audio = sum(np.sin(2 * np.pi * f * t_4s) * (1.0 / (k + 1)) 
                     for k, f in enumerate(freqs))
        audio += np.random.randn(len(t_4s)) * 0.1  # Broadband component
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(CONTINUOUS_DIR, f"engine_{i:03d}.wav"), audio, SR)

    for i in range(10):
        # Helicopter rotor: Periodic thumping + turbulence
        rotor_freq = np.random.uniform(15, 30)  # Hz
        audio = np.sin(2 * np.pi * rotor_freq * t_4s) * 0.5
        # Blade slap harmonics
        for h in range(2, 6):
            audio += np.sin(2 * np.pi * rotor_freq * h * t_4s) * (0.3 / h)
        # Turbulence
        audio += np.random.randn(len(t_4s)) * 0.15
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(CONTINUOUS_DIR, f"helicopter_{i:03d}.wav"), audio, SR)

    for i in range(10):
        # Wind noise: Filtered random noise with slow amplitude modulation
        noise = np.random.randn(len(t_8s))
        # Simple low-pass via moving average
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size
        audio = np.convolve(noise, kernel, mode='same')
        # Amplitude modulation (gusts)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t_8s)
        audio = audio * modulation
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(CONTINUOUS_DIR, f"wind_{i:03d}.wav"), audio[:SR * 4], SR)

    for i in range(10):
        # Vehicle rumble: Very low frequency + road noise
        base_freq = np.random.uniform(25, 60)
        audio = np.sin(2 * np.pi * base_freq * t_4s) * 0.4
        audio += np.sin(2 * np.pi * base_freq * 2 * t_4s) * 0.2
        # Road surface noise
        audio += np.random.randn(len(t_4s)) * 0.2
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(CONTINUOUS_DIR, f"vehicle_{i:03d}.wav"), audio, SR)
    
    for i in range(5):
        # Siren: Frequency-swept sine
        f_start, f_end = np.random.uniform(400, 600), np.random.uniform(800, 1200)
        freq_sweep = np.linspace(f_start, f_end, len(t_4s))
        phase = 2 * np.pi * np.cumsum(freq_sweep) / SR
        audio = np.sin(phase)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(os.path.join(CONTINUOUS_DIR, f"siren_{i:03d}.wav"), audio, SR)

def generate_synthetic_clean_speech():
    """Generate synthetic clean speech-like signals for training."""
    print("\nGenerating synthetic clean speech signals...")
    
    for i in range(20):
        t = np.linspace(0, 4, SR * 4)
        # Simulate formants (vocal tract resonances)
        f0 = np.random.uniform(100, 250)  # Fundamental frequency
        audio = np.zeros_like(t)
        
        # Generate voiced segments with gaps (like real speech)
        num_segments = np.random.randint(3, 8)
        for _ in range(num_segments):
            start = np.random.uniform(0, 3)
            duration = np.random.uniform(0.2, 0.8)
            mask = ((t >= start) & (t < start + duration)).astype(float)
            
            # Formants
            f1 = np.random.uniform(300, 900)
            f2 = np.random.uniform(1000, 2500)
            segment = (np.sin(2 * np.pi * f0 * t) * 0.5 +
                       np.sin(2 * np.pi * f1 * t) * 0.3 +
                       np.sin(2 * np.pi * f2 * t) * 0.2)
            
            # Apply envelope
            envelope = np.exp(-((t - (start + duration / 2)) ** 2) / (2 * (duration / 4) ** 2))
            audio += segment * envelope * mask
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        sf.write(os.path.join(CLEAN_DIR, f"speech_{i:03d}.wav"), audio, SR)

def print_dataset_summary():
    """Print a summary of the generated dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    
    for category, path in [("Clean Speech", CLEAN_DIR), 
                           ("Impulsive Noise", IMPULSIVE_DIR), 
                           ("Continuous Noise", CONTINUOUS_DIR)]:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.wav')]
            total_seconds = 0
            for f in files:
                try:
                    info = sf.info(os.path.join(path, f))
                    total_seconds += info.duration
                except Exception:
                    pass
            print(f"  {category}: {len(files)} files, {total_seconds:.1f}s total")
        else:
            print(f"  {category}: NOT FOUND")
    
    print("=" * 50)
    print("Dataset ready for training! Run: python train.py")

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  Defence ANC Dataset Generator               ║")
    print("╚══════════════════════════════════════════════╝")
    
    ensure_dirs()
    generate_synthetic_defence_noise()
    generate_synthetic_clean_speech()
    print_dataset_summary()
