#!/usr/bin/env python3
"""
Enhanced Solar Storm YOLO Trainer
Trains YOLO model on the new enhanced distinct dataset
"""

import os
from ultralytics import YOLO
import torch
from datetime import datetime
import yaml

class EnhancedSolarYOLOTrainer:
    def __init__(self):
        """Initialize the trainer"""
        self.dataset_path = "data/enhanced_distinct_dataset/yolo_format"
        self.model_output_dir = "models/enhanced_distinct_yolo"
        
        # Ensure output directory exists
        os.makedirs(self.model_output_dir, exist_ok=True)
        
    def check_dataset(self):
        """Verify dataset structure and configuration"""
        print("🔍 Checking dataset structure...")
        
        dataset_yaml = f"{self.dataset_path}/dataset.yaml"
        if not os.path.exists(dataset_yaml):
            print(f"❌ Dataset configuration not found: {dataset_yaml}")
            return False
        
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Dataset configuration found")
        print(f"  Classes: {config['names']}")
        print(f"  Number of classes: {config['nc']}")
        
        # Check if directories exist
        for split in ['train', 'val', 'test']:
            images_dir = f"{self.dataset_path}/{split}/images"
            labels_dir = f"{self.dataset_path}/{split}/labels"
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                num_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                print(f"  {split}: {num_images} images, {num_labels} labels")
            else:
                print(f"  ❌ {split} directory missing")
                return False
        
        return True
    
    def train_model(self, epochs=100, patience=20):
        """Train the YOLO model"""
        print(f"\n🚀 Starting YOLO training...")
        print(f"Epochs: {epochs}, Patience: {patience}")
        print(f"Output directory: {self.model_output_dir}")
        
        # Check CUDA availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training device: {device}")
        
        # Initialize model
        model = YOLO('yolov8m.pt')  # Use medium model for better accuracy
        
        # Training arguments
        train_args = {
            'data': f"{self.dataset_path}/dataset.yaml",
            'epochs': epochs,
            'patience': patience,
            'batch': 16 if device == 'cuda' else 8,
            'imgsz': 640,
            'device': device,
            'workers': 4,
            'project': self.model_output_dir,
            'name': 'enhanced_training',
            'save_period': 10,  # Save every 10 epochs
            'verbose': True,
            'plots': True,
            'val': True,
            'save': True,
            'exist_ok': True,
            
            # Optimization settings for better convergence
            'lr0': 0.01,          # Initial learning rate
            'momentum': 0.937,    # SGD momentum
            'weight_decay': 0.0005,  # Optimizer weight decay
            'warmup_epochs': 3,   # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup initial momentum
            'warmup_bias_lr': 0.1,   # Warmup initial bias lr
            'box': 7.5,           # Box loss gain
            'cls': 0.5,           # Class loss gain
            'dfl': 1.5,           # DFL loss gain
            
            # Data augmentation
            'hsv_h': 0.015,       # HSV-Hue augmentation
            'hsv_s': 0.7,         # HSV-Saturation augmentation
            'hsv_v': 0.4,         # HSV-Value augmentation
            'degrees': 10.0,      # Rotation degrees
            'translate': 0.1,     # Translation fraction
            'scale': 0.9,         # Scaling factor
            'shear': 0.0,         # Shear degrees
            'perspective': 0.0,   # Perspective factor
            'flipud': 0.0,        # Vertical flip probability
            'fliplr': 0.5,        # Horizontal flip probability
            'mosaic': 1.0,        # Mosaic probability
            'mixup': 0.1,         # Mixup probability
            'copy_paste': 0.0     # Copy paste probability
        }
        
        try:
            # Start training
            results = model.train(**train_args)
            
            print(f"\n✅ Training completed!")
            
            # Save the best model with a clear name
            best_model_path = f"{self.model_output_dir}/enhanced_training/weights/best.pt"
            if os.path.exists(best_model_path):
                final_model_path = f"{self.model_output_dir}/enhanced_distinct_best.pt"
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                print(f"✓ Best model saved to: {final_model_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def validate_model(self):
        """Validate the trained model"""
        print(f"\n🧪 Validating trained model...")
        
        model_path = f"{self.model_output_dir}/enhanced_distinct_best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        dataset_yaml = f"{self.dataset_path}/dataset.yaml"
        results = model.val(data=dataset_yaml, split='test')
        
        print(f"✅ Validation complete!")
        print(f"Model performance on test set:")
        
        # Extract key metrics
        if hasattr(results, 'box'):
            metrics = results.box
            if hasattr(metrics, 'map50'):
                print(f"  mAP@0.5: {metrics.map50:.3f}")
            if hasattr(metrics, 'map'):
                print(f"  mAP@0.5:0.95: {metrics.map:.3f}")
        
        return results
    
    def test_on_samples(self):
        """Test the model on sample images"""
        print(f"\n🖼️ Testing model on sample images...")
        
        model_path = f"{self.model_output_dir}/enhanced_distinct_best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        model = YOLO(model_path)
        
        # Test on sample images
        sample_dir = "data/enhanced_distinct_dataset/samples"
        if not os.path.exists(sample_dir):
            print(f"❌ Sample directory not found: {sample_dir}")
            return
        
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
        
        for sample_file in sample_files:
            sample_path = os.path.join(sample_dir, sample_file)
            
            # Run inference
            results = model(sample_path, verbose=False)
            
            print(f"\n  📸 {sample_file}:")
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    class_name = model.names[cls]
                    print(f"    ✓ {class_name}: {conf:.3f}")
            else:
                print(f"    ○ No detections")

def main():
    """Main execution function"""
    print("🌞 Enhanced Solar Storm YOLO Trainer")
    print("=" * 45)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = EnhancedSolarYOLOTrainer()
    
    # Step 1: Check dataset
    if not trainer.check_dataset():
        print("❌ Dataset check failed. Please fix the dataset first.")
        return
    
    # Step 2: Train model
    print(f"\n{'='*50}")
    print("🎯 TRAINING PHASE")
    print(f"{'='*50}")
    
    results = trainer.train_model(epochs=80, patience=15)
    
    if results is None:
        print("❌ Training failed. Exiting.")
        return
    
    # Step 3: Validate model
    print(f"\n{'='*50}")
    print("✅ VALIDATION PHASE")
    print(f"{'='*50}")
    
    trainer.validate_model()
    
    # Step 4: Test on samples
    print(f"\n{'='*50}")
    print("🧪 SAMPLE TESTING PHASE")
    print(f"{'='*50}")
    
    trainer.test_on_samples()
    
    print(f"\n🎉 All phases complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🎯 Next steps:")
    print("1. Check training results and plots")
    print("2. Update Streamlit app to use new model")
    print("3. Test improved storm detection accuracy")

if __name__ == "__main__":
    main()