"""
Fine-tune a Diffusion Model on Radio Galaxy Images
Trains on original + augmented images, outputs synthetic samples
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np


ORIGINAL_DIR = "exo/exo_PNG"
AUGMENTED_DIR = "exo_augmented"
OUTPUT_DIR = "exo_synthetic"
MODEL_SAVE_DIR = "diffusion_model"


IMAGE_SIZE = 128  
BATCH_SIZE = 2  
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_SYNTHETIC_IMAGES = 200 

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class GalaxyDataset(Dataset):
    """Dataset that loads from multiple directories"""
    
    def __init__(self, directories, image_size=128):
        self.image_paths = []
        for directory in directories:
            path = Path(directory)
            if path.exists():
                self.image_paths.extend(list(path.glob("*.png")))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        
        print(f"Loaded {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)


def train_diffusion_model():
    """Train/fine-tune diffusion model"""
    
    dataset = GalaxyDataset([ORIGINAL_DIR, AUGMENTED_DIR], IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    ).to(DEVICE)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * NUM_EPOCHS),
    )
    
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0
        
        for batch in progress_bar:
            clean_images = batch.to(DEVICE)
            
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],), device=DEVICE
            ).long()
            
            noise = torch.randn_like(clean_images)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 20 == 0:
            save_path = Path(MODEL_SAVE_DIR)
            save_path.mkdir(exist_ok=True)
            model.save_pretrained(save_path / f"checkpoint_epoch_{epoch+1}")
    
    save_path = Path(MODEL_SAVE_DIR)
    save_path.mkdir(exist_ok=True)
    model.save_pretrained(save_path / "final")
    noise_scheduler.save_pretrained(save_path / "final")
    print(f"\n✓ Model saved to {save_path / 'final'}")
    
    return model, noise_scheduler


def generate_synthetic_images(model, noise_scheduler, num_images):
    """Generate synthetic images using trained model"""
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline = pipeline.to(DEVICE)
    
    print(f"\nGenerating {num_images} synthetic images...")
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image = pipeline(batch_size=1, num_inference_steps=1000).images[0]
            
            image.save(output_path / f"synthetic_{i+1:04d}.png")
    
    print(f"✓ Synthetic images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    if not Path(ORIGINAL_DIR).exists():
        print(f"Error: {ORIGINAL_DIR} not found")
        exit(1)
    
    if not Path(AUGMENTED_DIR).exists():
        print(f"Warning: {AUGMENTED_DIR} not found, using only original images")
    
    model, scheduler = train_diffusion_model()
    
    generate_synthetic_images(model, scheduler, NUM_SYNTHETIC_IMAGES)
    
    print("\n✓ Complete!")

