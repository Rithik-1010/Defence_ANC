import torch
import torch.nn as nn

class TransientDetector(nn.Module):
    """
    Detects if the current frame contains impulsive noise (gunfire, explosion)
    or continuous noise (engine, wind).
    Takes spectrogram magnitude as input.
    """
    def __init__(self, n_freq_bins=257):
        super().__init__()
        # Input shape: (Batch, 1, Freq, Time)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1))
        
        # Adaptive pool to handle variable time lengths and pool them to 1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x expected to be (B, F, T) magnitude spectrogram
        x = x.unsqueeze(1) # (B, 1, F, T)
        
        out = self.relu(self.conv1(x))
        out = self.pool(out)
        out = self.relu(self.conv2(out))
        
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        prob = self.sigmoid(out)
        return prob.squeeze(1) # (B,) probability of impulsive

if __name__ == "__main__":
    detector = TransientDetector()
    # Batch=4, Freq=257, Time=100
    dummy_input = torch.abs(torch.randn(4, 257, 100, dtype=torch.complex64))
    prob = detector(dummy_input)
    print(f"Probabilities: {prob}")
