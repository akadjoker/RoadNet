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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jetson_road_detection.log')
    ]
)
logger = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser(description='Train optimized road detection model for Jetson Nano')
    parser.add_argument('--image_dir', type=str, default='fotos2', help='Directory with input images')
    parser.add_argument('--mask_dir', type=str, default='output_masks', help='Directory with mask images')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square) - reduced for speed')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size - reduced for Jetson memory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=78942, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='road_model', help='Output directory')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--export_tensorrt', action='store_true', help='Export model to TensorRT format')
    parser.add_argument('--quantize', action='store_true', help='Quantize model to int8')
    parser.add_argument('--road_focus', action='store_true', help='Focus on road features with data augmentation')
    parser.add_argument('--half_precision', action='store_true', help='Use FP16 (half precision) for training')
    return parser.parse_args()

# ==== DATASET - Optimized for road detection ====
class RoadDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(64, 64), augment=False, road_focus=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        self.road_focus = road_focus

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image with reduced size immediately
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask with reduced size
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        
        # Augmentations optimized for road detection
        if self.augment:
            # Horizontal flip (very fast)
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            
            # Brightness/contrast adjustments (roads in different lighting)
            if random.random() > 0.5:
                alpha = 0.8 + random.random() * 0.4  # 0.8-1.2
                beta = -10 + random.random() * 20  # -10 to 10
                img = np.clip(alpha * img + beta, 0, 255).astype(np.float32)
            
            # Add road-specific augmentations when road_focus is enabled
            if self.road_focus and random.random() > 0.7:
                # Simulate shadows or lighting changes on roads
                h, w, _ = img.shape
                y1, y2 = int(h * 0.4), int(h * 0.9)  # Focus on lower part where roads usually are
                x1, x2 = random.randint(0, w//2), random.randint(w//2, w)
                shadow_mask = np.ones_like(img) * 0.8
                img = np.clip(img * shadow_mask, 0, 255).astype(np.float32)
        
        # Normalize to [0,1] range
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW format
        
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0)

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

# ==== ROAD DETECTION FOCUSED LOSS FUNCTION ====
class RoadFocalLoss(nn.Module):
    """Focal Loss to focus more on hard examples, good for road/non-road imbalance"""
    def __init__(self, alpha=0.8, gamma=2.0):
        super(RoadFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# ==== ULTRA LIGHTWEIGHT MODEL ARCHITECTURE FOR JETSON ====
class UltraLightSeparableConv(nn.Module):
    """Highly optimized depthwise separable convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class JetsonRoadBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use single depthwise separable conv to reduce parameters
        self.conv = nn.Sequential(
            UltraLightSeparableConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class JetsonRoadNet(nn.Module):
    """Ultra lightweight UNet designed for road detection on Jetson Nano"""
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 48, 64]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part - Encoder
        prev_channels = in_channels
        for feature in features:
            self.downs.append(JetsonRoadBlock(prev_channels, feature))
            prev_channels = feature  # Atualiza os canais de saída para o próximo bloco
        
        # Bottleneck
        self.bottleneck = JetsonRoadBlock(features[-1], features[-1]*2)
        prev_channels = features[-1]*2  # Atualiza canais para o decoder (começa com 128)

        # Up part - Decoder com skip connections
        for feature in reversed(features):
            # O upsampling precisa saber de onde vem: prev_channels
            self.ups.append(
                nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2)
            )
            self.ups.append(JetsonRoadBlock(feature*2, feature))
            prev_channels = feature  # Atualiza para o próximo loop
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Road attention - ajuda a focar nas features de estrada
        self.road_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(features[0], features[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            upconv = self.ups[idx]
            convblock = self.ups[idx + 1]
            x = upconv(x)
            skip_connection = skip_connections[idx // 2]
            
            # Corrige mismatch de tamanho se existir
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = convblock(concat_skip)

        # Road attention
        attention = self.road_attention(x)
        x = x * attention

        return torch.sigmoid(self.final_conv(x))

        
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
        
        # Apply road attention
        if x.shape[0] > 0:  # Check if batch isn't empty
            attention = self.road_attention(x)
            x = x * attention
        
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
            return False
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
    train_dataset = RoadDataset(
        train_imgs, train_masks, img_size=(args.img_size, args.img_size), 
        augment=True, road_focus=args.road_focus
    )
    val_dataset = RoadDataset(
        val_imgs, val_masks, img_size=(args.img_size, args.img_size)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=True  # Reduced workers for Jetson Nano
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        num_workers=2, pin_memory=True
    )
    
    # Initialize lightweight model
    model = JetsonRoadNet().to(device)
    
    # Use half precision if requested (FP16) - speeds up training on Jetson
    if args.half_precision and device.type == 'cuda':
        model = model.half()
        logger.info("Using half precision (FP16) for training")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Optimizers and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Choose loss function based on road focus parameter
    if args.road_focus:
        loss_fn = RoadFocalLoss()
        logger.info("Using Focal Loss for road detection focus")
    else:
        loss_fn = nn.BCELoss()
        logger.info("Using standard BCE Loss")
    
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
    logger.info(f"Starting Jetson road detection model training at {start_time}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train")
        for batch_idx, (images, masks) in enumerate(train_loop):
            # Move to device and convert to half precision if needed
            images, masks = images.to(device), masks.to(device)
            if args.half_precision and device.type == 'cuda':
                images = images.half()
            
            # Forward pass
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            dice = dice_metric(predictions, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val")
            for batch_idx, (images, masks) in enumerate(val_loop):
                images, masks = images.to(device), masks.to(device)
                if args.half_precision and device.type == 'cuda':
                    images = images.half()
                
                # Forward pass
                predictions = model(images)
                loss = loss_fn(predictions, masks)
                dice = dice_metric(predictions, masks)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), dice=dice.item())
        
        # Calculate average metrics for the epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
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
    
    # Export to ONNX if requested
    if args.export_onnx:
        onnx_path = export_to_onnx(model, args)
        
        # Export to TensorRT if requested (requires ONNX)
        if args.export_tensorrt:
            export_to_tensorrt(onnx_path, args)
    
    # Quantize model if requested
    if args.quantize:
        quantize_model(model, val_loader, device, args)
    
    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, train_dice_scores, val_dice_scores, args.output_dir)
    
    # Return best validation dice score for reporting
    return max(val_dice_scores)

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
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    
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

def export_to_onnx(model, args):
    """Export model to ONNX format for inference optimization"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=next(model.parameters()).device)
    
    # Export path
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    
    logger.info(f"Model exported to ONNX format at {onnx_path}")
    return onnx_path

def export_to_tensorrt(onnx_path, args):
    """Export ONNX model to TensorRT for faster inference on Jetson"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger.info("Converting ONNX model to TensorRT...")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # Configure builder
            builder.max_workspace_size = 1 << 28  # 256 MiB
            builder.max_batch_size = 1
            builder.fp16_mode = args.half_precision
            
            # Parse ONNX
            with open(onnx_path, 'rb') as model_file:
                parser.parse(model_file.read())
            
            # Build engine
            engine = builder.build_cuda_engine(network)
            
            # Serialize engine
            with open(os.path.join(args.output_dir, "model.trt"), "wb") as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to {os.path.join(args.output_dir, 'model.trt')}")
    
    except ImportError:
        logger.warning("TensorRT export requires the TensorRT package to be installed.")
    except Exception as e:
        logger.error(f"TensorRT conversion failed: {e}")

def quantize_model(model, dataloader, device, args):
    """Quantize model to int8 for better performance on Jetson"""
    try:
        import torch.quantization
        
        logger.info("Starting model quantization...")
        
        # Set model to eval mode
        model.eval()
        
        # Move model to CPU for quantization
        model.to('cpu')
        
        # Prepare model for static quantization
        model_fp32_prepared = torch.quantization.prepare(model)
        
        # Calibrate with the first few batches
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                model_fp32_prepared(images.to('cpu'))
                if i >= 10:  # Calibrate on 10 batches
                    break
        
        # Convert to quantized model
        model_int8 = torch.quantization.convert(model_fp32_prepared)
        
        # Save quantized model
        quantized_path = os.path.join(args.output_dir, "quantized_model.pth")
        torch.save(model_int8.state_dict(), quantized_path)
        logger.info(f"Quantized model saved to {quantized_path}")
        
        # Additional path for easy TensorRT conversion
        torch.save({
            'state_dict': model_int8.state_dict(),
            'img_size': args.img_size,
        }, os.path.join(args.output_dir, "quantized_model_full.pth"))
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")

def inference_jetson(image_path, model_path, img_size=64):
    """Helper function for inference on Jetson Nano"""
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JetsonRoadNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)
    
    # Post-process prediction
    pred_np = pred.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.5).astype(np.uint8) * 255
    
    # Resize back to original size
    h, w = img.shape[:2]
    pred_resized = cv2.resize(pred_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return pred_resized

def main():
    args = parse_args()
    best_dice = train(args)
    logger.info(f"Training complete with best Dice score: {best_dice:.4f}")


if __name__ == "__main__":
    main()
