#!/usr/bin/env python3
"""
Train Vision Transformer (ViT) for solar storm classification
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import glob
import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from cv_pipeline.config import (VIT_DATASET_DIR, VIT_MODEL_NAME, VIT_IMAGE_SIZE, 
                                VIT_EPOCHS, VIT_BATCH_SIZE, VIT_LEARNING_RATE, VIT_CLASSES)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarImageDataset(Dataset):
    """Dataset for solar images with storm/quiet classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SolarViTTrainer:
    """Train Vision Transformer for solar storm classification"""
    
    def __init__(self, model_name: str = VIT_MODEL_NAME, num_classes: int = len(VIT_CLASSES)):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = os.path.join("models", "vit")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized ViT trainer with model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Initialize model
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create training and validation transforms"""
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_dummy_dataset(self) -> Tuple[List[str], List[int]]:
        """Create dummy dataset from existing images with synthetic labels"""
        from cv_pipeline.config import YOLO_DATASET_DIR
        
        # Look for images in YOLO dataset
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        all_images = []
        
        for split in ["train", "val", "test"]:
            images_dir = os.path.join(YOLO_DATASET_DIR, "images", split)
            if os.path.exists(images_dir):
                for ext in image_extensions:
                    pattern = os.path.join(images_dir, ext)
                    all_images.extend(glob.glob(pattern))
        
        if not all_images:
            logger.warning("No images found for ViT training")
            return [], []
        
        # Create synthetic labels based on filename patterns or random assignment
        image_paths = []
        labels = []
        
        for img_path in all_images:
            image_paths.append(img_path)
            
            # Synthetic labeling logic
            basename = os.path.basename(img_path).lower()
            if any(indicator in basename for indicator in ['193', '304', 'flare', 'storm']):
                labels.append(1)  # Storm
            else:
                labels.append(0)  # Quiet
        
        logger.info(f"Created dataset with {len(image_paths)} images")
        storm_count = sum(labels)
        quiet_count = len(labels) - storm_count
        logger.info(f"Storm images: {storm_count}, Quiet images: {quiet_count}")
        
        return image_paths, labels
    
    def prepare_data_loaders(self, test_split: float = 0.2, val_split: float = 0.1):
        """Prepare train/val/test data loaders"""
        
        # Get dataset
        image_paths, labels = self.create_dummy_dataset()
        
        if not image_paths:
            logger.error("No images available for training")
            return False
        
        # Split data
        total_samples = len(image_paths)
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * val_split)
        train_size = total_samples - test_size - val_size
        
        # Create splits
        train_paths = image_paths[:train_size]
        train_labels = labels[:train_size]
        
        val_paths = image_paths[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        
        test_paths = image_paths[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
        
        logger.info(f"Data split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Create transforms
        train_transform, val_transform = self.create_transforms()
        
        # Create datasets
        train_dataset = SolarImageDataset(train_paths, train_labels, train_transform)
        val_dataset = SolarImageDataset(val_paths, val_labels, val_transform)
        test_dataset = SolarImageDataset(test_paths, test_labels, val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=VIT_BATCH_SIZE, shuffle=True, 
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=VIT_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=VIT_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True if torch.cuda.is_available() else False
        )
        
        return True
    
    def create_model(self):
        """Create Vision Transformer model"""
        logger.info(f"Creating model: {self.model_name}")
        
        # Load pretrained model
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.num_classes
        )
        
        self.model = self.model.to(self.device)
        logger.info(f"Model created and moved to {self.device}")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs: int = VIT_EPOCHS, lr: float = VIT_LEARNING_RATE):
        """Train the ViT model"""
        
        if not self.prepare_data_loaders():
            logger.error("Failed to prepare data loaders")
            return None
        
        self.create_model()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        best_val_acc = 0.0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch + 1)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Log progress
            logger.info(f'Epoch {epoch+1}/{epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.output_dir, "solar_vit_best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, model_path)
                logger.info(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "solar_vit_final.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, final_model_path)
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
        
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history
    
    def test(self, model_path: str = None):
        """Test the trained model"""
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        
        if self.model is None:
            logger.error("No model loaded for testing")
            return None
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate classification report
        class_names = list(VIT_CLASSES.values())
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        logger.info("Classification Report:")
        logger.info(f"\n{report}")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Save results
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        results_path = os.path.join(self.output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {results_path}")
        return results


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ViT model for solar storm classification")
    parser.add_argument("--model", default=VIT_MODEL_NAME, help=f"Model name (default: {VIT_MODEL_NAME})")
    parser.add_argument("--epochs", type=int, default=VIT_EPOCHS, help=f"Number of epochs (default: {VIT_EPOCHS})")
    parser.add_argument("--batch", type=int, default=VIT_BATCH_SIZE, help=f"Batch size (default: {VIT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=VIT_LEARNING_RATE, help=f"Learning rate (default: {VIT_LEARNING_RATE})")
    parser.add_argument("--test", action="store_true", help="Run test after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SolarViTTrainer(model_name=args.model)
    
    # Train model
    history = trainer.train(epochs=args.epochs, lr=args.lr)
    
    if history is None:
        logger.error("Training failed")
        return 1
    
    # Test if requested
    if args.test:
        best_model_path = os.path.join(trainer.output_dir, "solar_vit_best.pth")
        trainer.test(best_model_path)
    
    logger.info("ViT training pipeline completed!")
    return 0


if __name__ == "__main__":
    exit(main())