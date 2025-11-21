import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
from torchvision.models.vgg import vgg16

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------- MODEL DEFINITION ----------------

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma * out + x
        return out

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out) + x)
        return out

class Cascading_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.r1, self.r2, self.r3 = Residual_Block(channels), Residual_Block(channels), Residual_Block(channels)
        self.c1, self.c2, self.c3 = Basic_Block(channels * 2, channels), Basic_Block(channels * 3, channels), Basic_Block(channels * 4, channels)
        self.attn = SelfAttention(channels)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.r1(o0)
        c1, o1 = torch.cat([c0, b1], dim=1), self.c1(torch.cat([c0, b1], dim=1))
        b2 = self.r2(o1)
        c2, o2 = torch.cat([c1, b2], dim=1), self.c2(torch.cat([c1, b2], dim=1))
        b3 = self.r3(o2)
        c3, o3 = torch.cat([c2, b3], dim=1), self.c3(torch.cat([c2, b3], dim=1))
        return self.attn(o3)

class ScHiCAtt(nn.Module):
    def __init__(self, num_channels=64):
        super().__init__()
        self.entry = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.cb1, self.cb2, self.cb3, self.cb4, self.cb5 = Cascading_Block(num_channels), Cascading_Block(num_channels), Cascading_Block(num_channels), Cascading_Block(num_channels), Cascading_Block(num_channels)
        self.cv1, self.cv2, self.cv3, self.cv4, self.cv5 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=1), nn.Conv2d(num_channels * 3, num_channels, kernel_size=1), nn.Conv2d(num_channels * 4, num_channels, kernel_size=1), nn.Conv2d(num_channels * 5, num_channels, kernel_size=1), nn.Conv2d(num_channels * 6, num_channels, kernel_size=1)
        self.exit = nn.Conv2d(num_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.entry(x)
        c0 = o0 = x
        b1, c1, o1 = self.cb1(o0), torch.cat([c0, self.cb1(o0)], dim=1), self.cv1(torch.cat([c0, self.cb1(o0)], dim=1))
        b2, c2, o2 = self.cb2(o1), torch.cat([c1, self.cb2(o1)], dim=1), self.cv2(torch.cat([c1, self.cb2(o1)], dim=1))
        b3, c3, o3 = self.cb3(o2), torch.cat([c2, self.cb3(o2)], dim=1), self.cv3(torch.cat([c2, self.cb3(o2)], dim=1))
        b4, c4, o4 = self.cb4(o3), torch.cat([c3, self.cb4(o3)], dim=1), self.cv4(torch.cat([c3, self.cb4(o3)], dim=1))
        b5, c5, o5 = self.cb5(o4), torch.cat([c4, self.cb5(o4)], dim=1), self.cv5(torch.cat([c4, self.cb5(o4)], dim=1))
        return self.exit(o5)

# ---------------- LOSS FUNCTION ----------------

class TVLoss(nn.Module):
    def forward(self, x):
        return ((x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum() + (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()) / x.shape[0]

class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:31].eval()
        for param in vgg.parameters(): param.requires_grad = False
        self.loss_network = vgg
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images.repeat([1, 3, 1, 1])).view(out_images.size(0), -1), self.loss_network(target_images.repeat([1, 3, 1, 1])).view(target_images.size(0), -1))
        image_loss = self.mse_loss(out_images, target_images)
        return image_loss + 0.001 * perception_loss + 2e-8 * self.tv_loss(out_images)

# ---------------- DATA LOADER ----------------

def load_dataset(npz_path):
    data = np.load(npz_path)
    return TensorDataset(
        torch.tensor(data['data'], dtype=torch.float32),
        torch.tensor(data['target'], dtype=torch.float32)
    )

# ---------------- TRAINING FUNCTION ----------------

def train_schicatt(train_data_path, valid_data_path, num_epochs=1, batch_size=64, lr=0.0003, save_path="schicatt.pth"):
    abs_train_path = os.path.abspath(train_data_path)
    abs_valid_path = os.path.abspath(valid_data_path)

    print(f"Looking for training data at: {abs_train_path}")
    print(f"Looking for validation data at: {abs_valid_path}")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found: {abs_train_path}")
    if not os.path.exists(valid_data_path):
        raise FileNotFoundError(f"Validation data not found: {abs_valid_path}")

    print("Loading data...")
    train_loader = DataLoader(load_dataset(train_data_path), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(load_dataset(valid_data_path), batch_size=batch_size, shuffle=False)

    model = ScHiCAtt(num_channels=64).to(device)
    criterion = GeneratorLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, target = data.to(device), target.to(device)
            # Light training augmentation
            if model.training:
                # Small Gaussian noise
                noise = torch.randn_like(data) * 0.02
                data = data + noise

                # Slight value scaling
                if torch.rand(1).item() > 0.5:
                    factor = torch.empty(1).uniform_(0.9, 1.1).item()
                    data = data * factor

                # Randomly drop a few entries , simulates missing contacts
                if torch.rand(1).item() > 0.5:
                    mask = (torch.rand_like(data) > 0.98).float()
                    data = data * (1 - mask)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_data, val_target in valid_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_output = model(val_data)
                val_loss += criterion(val_output, val_target).item()
            avg_val_loss = val_loss / len(valid_loader)
            print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")

# ---------------- RUN TRAINING ----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ScHiCAtt model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training dataset (.npz)")
    parser.add_argument("--valid_data", type=str, required=True, help="Path to validation dataset (.npz)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="schicatt.pth", help="Path to save the trained model")

    args = parser.parse_args()

    train_schicatt(args.train_data, args.valid_data, args.epochs, args.batch_size, args.lr, args.save_path)
