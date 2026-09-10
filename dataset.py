import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class ANCDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sr=16000, chunk_length_s=4.0, snr_range=(-5, 15)):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.sr = sr
        self.chunk_samples = int(sr * chunk_length_s)
        self.snr_range = snr_range
        
        # Load file paths safely (ensure directories exist)
        self.clean_files = self._get_files(clean_dir) if os.path.exists(clean_dir) else []
        self.noise_files = self._get_files(noise_dir) if os.path.exists(noise_dir) else []
        
        # Mock mode if directories are empty (for testing pipeline)
        self.mock_mode = len(self.clean_files) == 0 or len(self.noise_files) == 0
        if self.mock_mode:
            print("Warning: Dataset directories empty or missing. Using mock data generation for pipeline validation.")

    def _get_files(self, d):
        return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.wav')]
        
    def _load_and_pad(self, path):
        # Load audio safely
        try:
            wav, sr = torchaudio.load(path)
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                wav = resampler(wav)
            wav = wav[0] # To mono
            
            # Pad or trim to chunk_samples
            if len(wav) > self.chunk_samples:
                start = random.randint(0, len(wav) - self.chunk_samples)
                wav = wav[start:start + self.chunk_samples]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.chunk_samples - len(wav)))
            return wav
        except Exception:
            return torch.zeros(self.chunk_samples)
            
    def __len__(self):
        return 100 if self.mock_mode else len(self.clean_files) * 10
        
    def __getitem__(self, idx):
        if self.mock_mode:
            clean = torch.randn(self.chunk_samples)
            noise = torch.randn(self.chunk_samples)
            noise_category = 0 # 0: continuous, 1: impulsive
        else:
            clean_path = random.choice(self.clean_files)
            noise_path = random.choice(self.noise_files)
            clean = self._load_and_pad(clean_path)
            noise = self._load_and_pad(noise_path)
            noise_category = 1 if 'impulsive' in noise_path.lower() else 0
            
        # Mix with random SNR
        snr = random.uniform(*self.snr_range)
        clean_rms = torch.sqrt(torch.mean(clean**2) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise**2) + 1e-8)
        
        desired_noise_rms = clean_rms / (10 ** (snr / 20))
        noise = noise * (desired_noise_rms / noise_rms)
        
        mixture = clean + noise
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(mixture))
        if max_val > 1.0:
            mixture /= max_val
            clean /= max_val
            
        return mixture, clean, snr, noise_category

if __name__ == "__main__":
    ds = ANCDataset("data/clean", "data/noise")
    mix, clean, snr, cat = ds[0]
    print(f"Sample output: mix shape {mix.shape}, snr {snr:.2f}, category {cat}")
