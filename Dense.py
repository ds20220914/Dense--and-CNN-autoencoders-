import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from PIL import Image
import os




class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim1=256, hidden_dim2=128, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat




# TRAINING THE MODEL    


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

            # ---- FLATTEN FOR DENSE MODEL ----
            x_flat = x.view(x.size(0), -1)

            optimizer.zero_grad()

            x_hat = model(x_flat)

            loss = F.mse_loss(x_hat, x_flat)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)

        print(
            f"DENSE (CPU) epoch {epoch+1}/{epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"time: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start

    print("\n--- DENSE CPU TRAINING DONE ---")
    print(f"Total training time: {total_time:.2f} s")
    print(f"Final loss: {avg_loss:.4f}")


# SAVE IMAGE RECONSTRUCTED IMAGE FROM TESTING


def save_image(tensor, path):
    img = tensor.detach().cpu().view(28, 28)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).byte().numpy()

    img = Image.fromarray(img, mode="L")
    img = img.resize((256, 256))

    img.save(path)




# TESTING THE MODEL AND SAVING THE RECONSTRUCTED IMAGES


def test(model, loader, device, save_dir="./Dense_images", n=10):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    criterion = nn.MSELoss()

    # ---- Data ----
    x, _ = next(iter(loader))
    x = x[:n].to(device)

    x_flat = x.view(n, -1)

    # ---- Measure compute time ----
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        x_hat = model(x_flat)
        loss = criterion(x_hat, x_flat)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_time = end_time - start_time

    # ---- Memory transfer estimation ----
    # Approx: bytes moved = input + output tensors (GPU)
    bytes_in = x.element_size() * x.nelement()
    bytes_out = x_hat.element_size() * x_hat.nelement()
    total_bytes = bytes_in + bytes_out

    # convert to MB
    total_mb = total_bytes / (1024 ** 2)

    # ---- Save images ----
    x_hat_img = x_hat.view_as(x)  # reshape back if needed

    for i in range(n):
        save_image(x[i], f"{save_dir}/original_{i}.jpg")
        save_image(x_hat_img[i], f"{save_dir}/recon_{i}.jpg")

    print(f"\n--- TEST METRICS OF DENSE AUTOENCODER ---")
    print(f"Loss (MSE): {loss.item():.6f}")
    print(f"Inference time: {inference_time:.6f} s")
    print(f"Approx memory traffic: {total_mb:.2f} MB")
    print(f"Saved {n} reconstructed images to {save_dir}")

