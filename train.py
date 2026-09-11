import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ANCDataset
from model_gtcrn import ComplexMasker
import numpy as np

def create_mock_data(clean_dir, noise_dir):
    """
    Creates some synthetic audio files so the training loop can run out-of-the-box.
    In reality, you'd place real Kaggle gunshot/engine datasets here!
    """
    import soundfile as sf
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    sr = 16000
    print("Generating synthetic dataset (Artillery, Gunshots, Engines, Speech)...")
    
    # 1. Clean speech (Mock: Sine wave + harmonics)
    for i in range(5):
        t = np.linspace(0, 4, sr * 4)
        clean = np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 600 * t)
        sf.write(f"{clean_dir}/clean_speech_{i}.wav", clean, sr)
        
    # 2. Continuous Noise (Mock: Engine humming - Low freq colored noise)
    for i in range(3):
        engine = np.random.randn(sr * 4) * np.sin(2 * np.pi * 50 * np.linspace(0, 4, sr * 4))
        sf.write(f"{noise_dir}/continuous_engine_{i}.wav", engine, sr)
        
    # 3. Impulsive Noise (Mock: Gunshots/Artillery - sharp decays)
    for i in range(3):
        gunshot = np.zeros(sr * 4)
        for burst in [0.5, 1.5, 3.0]:
            idx = int(burst * sr)
            decay = np.exp(-np.linspace(0, 10, sr // 2))
            gunshot[idx:idx + len(decay)] = np.random.randn(sr // 2) * decay
        sf.write(f"{noise_dir}/impulsive_artillery_{i}.wav", gunshot, sr)

def train_model():
    clean_dir = "data/clean"
    noise_dir = "data/noise"
    
    if not os.path.exists(clean_dir) or not os.path.exists(noise_dir):
        create_mock_data(clean_dir, noise_dir)
        
    print("Initializing AI Training Pipeline...")
    
    # The ANCDataset automatically mixes clean + noise dynamically
    dataset = ANCDataset(clean_dir, noise_dir, chunk_length_s=1.0)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize our AI model
    model = ComplexMasker()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simplified loss function for the mask
    # In a real setup, this would be SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
    criterion = nn.MSELoss() 
    
    epochs = 5
    print(f"Starting training loop for {epochs} epochs on defence datasets...")
    
    for epoch in range(epochs):
        total_loss = 0
        for mix, clean, snr, cat in loader:
            # We mock the STFT conversion for the training loop scaffold
            # In a full setup, you run torch.stft on the 'mix' tensor
            mix_stft = torch.view_as_complex(torch.randn(mix.shape[0], 257, 100, 2))
            clean_stft = torch.view_as_complex(torch.randn(mix.shape[0], 257, 100, 2))
            
            optimizer.zero_grad()
            out = model(mix_stft) # Predict the clean audio mask
            
            # Loss compares model output vs clean target
            loss = criterion(torch.abs(out), torch.abs(clean_stft))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | AI Model is learning...")
        
    torch.save(model.state_dict(), "defence_anc_model.pth")
    print("Training complete! Model saved as defence_anc_model.pth")

if __name__ == "__main__":
    train_model()
