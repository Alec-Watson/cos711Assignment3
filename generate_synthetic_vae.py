"""
Train a VAE on Radio Galaxy Images and Generate Synthetic Samples
Uses Variational Autoencoder for image generation
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np



INPUT_DIR = "exo/exo_PNG"
OUTPUT_DIR = "exo_vae_synthetic"
MODEL_SAVE_PATH = "vae_model.pth"


IMAGE_SIZE = 128  
LATENT_DIM = 128 
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
NUM_SYNTHETIC_IMAGES = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class GalaxyDataset(Dataset):
    """Dataset for galaxy images"""
    
    def __init__(self, directory, image_size=128):
        self.image_paths = list(Path(directory).glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        print(f"Loaded {len(self.image_paths)} images from {directory}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)


class VAE(nn.Module):
    """Variational Autoencoder"""
    
    def __init__(self, image_size=128, latent_dim=128):
        super(VAE, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        self.encoded_size = image_size // 16 
        self.encoded_features = 256 * self.encoded_size * self.encoded_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
        )
        

        self.fc_mu = nn.Linear(self.encoded_features, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_features, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.encoded_features)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        """Encode input to latent space"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, self.encoded_size, self.encoded_size)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = Reconstruction loss + KL divergence"""

    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss


def train_vae():
    """Train the VAE model"""
    

    dataset = GalaxyDataset(INPUT_DIR, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    

    model = VAE(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nTraining VAE for {NUM_EPOCHS} epochs...")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Latent dimension: {LATENT_DIM}")
    

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in progress_bar:
            images = batch.to(DEVICE)
            

            recon_images, mu, logvar = model(images)
            loss = vae_loss(recon_images, images, mu, logvar)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item() / len(images)})
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
  
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"vae_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    

    torch.save({
        'model_state_dict': model.state_dict(),
        'image_size': IMAGE_SIZE,
        'latent_dim': LATENT_DIM,
    }, MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")
    
    return model


def generate_synthetic_images(model, num_images):
    """Generate synthetic images by sampling from latent space"""
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nGenerating {num_images} synthetic images...")
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_images)):

            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            

            generated = model.decode(z)
            

            img_tensor = generated[0].cpu()
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            

            img.save(output_path / f"vae_synthetic_{i+1:04d}.png")
    
    print(f"✓ Synthetic images saved to {OUTPUT_DIR}")


def generate_interpolations(model, num_interpolations=10):
    """Generate interpolations between two random points in latent space"""
    
    output_path = Path(OUTPUT_DIR) / "interpolations"
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\nGenerating {num_interpolations} interpolations...")
    
    model.eval()
    with torch.no_grad():

        z1 = torch.randn(1, LATENT_DIM).to(DEVICE)
        z2 = torch.randn(1, LATENT_DIM).to(DEVICE)
        
        for i in range(num_interpolations):
            alpha = i / (num_interpolations - 1)
            z = (1 - alpha) * z1 + alpha * z2
            
            generated = model.decode(z)
            
            img_tensor = generated[0].cpu()
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            img.save(output_path / f"interp_{i+1:02d}.png")
    
    print(f"✓ Interpolations saved to {output_path}")


if __name__ == "__main__":

    if not Path(INPUT_DIR).exists():
        print(f"Error: {INPUT_DIR} not found")
        exit(1)
    

    model = train_vae()
    

    generate_synthetic_images(model, NUM_SYNTHETIC_IMAGES)
    
    generate_interpolations(model, num_interpolations=10)
    
    print("\n✓ Complete!")

