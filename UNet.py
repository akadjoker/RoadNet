import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import random
import logging
from datetime import datetime
from torchvision import transforms
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

transform_augment = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.1
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(3)], p=0.1
    ),
    transforms.RandomApply(
        [transforms.Lambda(
            lambda img: img + 0.02 * torch.randn_like(img)
        )], p=0.2  # Ruído gaussiano leve
    )
])

def set_seed(seed=78942):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train U-Net for image segmentation')
    parser.add_argument('--image_dir', type=str, default='images', help='Directory with input images')
    parser.add_argument('--mask_dir', type=str, default='masks', help='Directory with mask images')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=9, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=78942, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

# ==== DATASET ====
class LineDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256), transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.augment:

            # Flip horizontal
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            if random.random() > 0.5:
                alpha = 0.8 + random.random() * 0.4  # 0.8-1.2
                beta = -10 + random.random() * 20  # -10 to 10
                img = np.clip(alpha * img + beta, 0, 255).astype(np.float32)
  

            # Color jitter (brilho/contraste/saturação/hue)
            img = transform_augment(torch.from_numpy(img).permute(2,0,1).float()/255.0)
            img = (img * 255).permute(1,2,0).byte().numpy()

            # Zoom in leve (crop e resize)
            if random.random() > 0.3:
                h, w, _ = img.shape
                crop = random.uniform(0.9, 1.0)
                nh, nw = int(h * crop), int(w * crop)
                top = random.randint(0, h - nh)
                left = random.randint(0, w - nw)
                img = img[top:top+nh, left:left+nw]
                mask = mask[top:top+nh, left:left+nw]
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size)


            


        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ==== METRICS ====
class DiceScore(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceScore, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        return (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth)

def iou_score(pred, target, smooth=1.0):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)

# ==== MODEL ====
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle cases where dimensions don't exactly match
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
                return True
        return False

def train(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get data paths
    img_paths = sorted(glob(os.path.join(args.image_dir, '*')))
    mask_paths = sorted(glob(os.path.join(args.mask_dir, '*')))
    
    if len(img_paths) == 0 or len(mask_paths) == 0:
        logger.error(f"No images found in {args.image_dir} or masks in {args.mask_dir}")
        return
    
    logger.info(f"Found {len(img_paths)} images and {len(mask_paths)} masks")
    
    # Split data
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=args.seed
    )
    
    # Create dataloaders
    train_dataset = LineDataset(
        train_imgs, train_masks, img_size=(args.img_size, args.img_size), augment=True
    )
    val_dataset = LineDataset(
        val_imgs, val_masks, img_size=(args.img_size, args.img_size)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )
    
    # Initialize model, optimizer, scheduler, and loss function
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_fn = nn.BCELoss()
    dice_metric = DiceScore()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        path=os.path.join(args.output_dir, 'best_model.pth')
    )
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    
    # Training loop
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train")
        for batch_idx, (images, masks) in enumerate(train_loop):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            dice = dice_metric(predictions, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_dice += dice.item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate average metrics for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        train_losses.append(avg_train_loss)
        train_dice_scores.append(avg_train_dice)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val")
            for batch_idx, (images, masks) in enumerate(val_loop):
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                predictions = model(images)
                loss = loss_fn(predictions, masks)
                dice = dice_metric(predictions, masks)
                batch_iou = iou_score(predictions, masks)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()
                val_iou += batch_iou
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate average metrics for the epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")
        
        # Save sample predictions
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_predictions(model, val_loader, device, epoch, args.output_dir)
        
        # Check early stopping
        if early_stopping(avg_val_loss, model):
            logger.info("Early stopping triggered")
            break
    
    # Training complete
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed in {training_duration}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, args.output_dir)

def save_predictions(model, dataloader, device, epoch, output_dir):
    """Save sample predictions for visualization"""
    model.eval()
    # Get a batch of validation data
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    # Save the first 4 samples or fewer if batch size is smaller
    num_samples = min(4, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")
        
        # Ground truth mask
        mask = masks[i, 0].cpu().numpy()
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        # Predicted mask
        pred = predictions[i, 0].cpu().numpy()
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"predictions_epoch_{epoch+1}.png"))
    plt.close()

def plot_training_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, output_dir):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot Dice scores
    plt.subplot(1, 2, 2)
    plt.plot(train_dice_scores, label="Train Dice")
    plt.plot(val_dice_scores, label="Val Dice")
    plt.title("Dice Score Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()   
    train(args)
