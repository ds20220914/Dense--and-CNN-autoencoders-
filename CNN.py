import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from PIL import Image
import os



# CNN AUTOENCODER

class CNNautoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODER (idea of CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # input 1 black white image, output 16 feature maps, scan with 3x3 kernel -> 28x28 -> 14x14 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # 7x7 -> 1x1
        )

        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # opposite of Conv2d
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat



# TRAINING LOOP

def train(model, loader, device, epochs=5, lr=1e-3):
    # pakotetaan CPU
    device = torch.device("cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0

        for x, _ in loader:
            x = x.to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = F.mse_loss(x_hat, x)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)

        print(
            f"CNN (CPU) epoch {epoch+1}/{epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"time: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start

    print("\n--- CNN CPU TRAINING DONE ---")
    print(f"Total training time: {total_time:.2f} s")
    print(f"Final loss: {avg_loss:.4f}")





# SAVE RECONSTRUCTED IMAGE FROM TESTING

def save_image(tensor, path):
    img = tensor.detach().cpu().squeeze(0)  # (1,28,28) -> (28,28)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).byte().numpy()

    img = Image.fromarray(img, mode="L")
    img = img.resize((256, 256))
    img.save(path)



# TEST FUNCTION

def test(model, loader, device, save_dir="./Dense_images", n=10):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()

    criterion = nn.MSELoss()

    # ---- Data ----
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    # ---- Measure compute time (CPU safe) ----
    start_time = time.time()

    with torch.no_grad():
        x_hat = model(x)
        loss = criterion(x_hat, x)

    end_time = time.time()

    inference_time = end_time - start_time

    # ---- Memory estimate (CPU-friendly approximation) ----
    bytes_in = x.element_size() * x.nelement()
    bytes_out = x_hat.element_size() * x_hat.nelement()
    total_mb = (bytes_in + bytes_out) / (1024 ** 2)

    # ---- Save images ----
    for i in range(n):
        save_image(x[i], f"{save_dir}/original_{i}.jpg")
        save_image(x_hat[i], f"{save_dir}/recon_{i}.jpg")

    print(f"\n--- TEST METRICS (CPU) ---")
    print(f"Loss (MSE): {loss.item():.6f}")
    print(f"Inference time: {inference_time:.6f} s")
    print(f"Approx memory traffic: {total_mb:.2f} MB")
    print(f"Saved {n} reconstructed images to {save_dir}")
