#!/usr/bin/env python3
"""
Enhanced Solar Dataset Creator
Creates a more visually distinct dataset for better storm vs quiet sun classification
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import json
from datetime import datetime
import shutil

class EnhancedSolarDatasetCreator:
    def __init__(self):
        """Initialize the enhanced dataset creator"""
        self.output_dir = "data/enhanced_distinct_dataset"
        self.yolo_dir = f"{self.output_dir}/yolo_format"
        
        # Class definitions with more distinct characteristics
        self.classes = {
            'quiet_sun': 0,
            'active_storm': 1,  # More active/bright regions
            'intense_flare': 2,  # Very bright localized regions
            'cme_activity': 3   # Large-scale disturbances
        }
        
    def create_directory_structure(self):
        """Create the directory structure for YOLO format"""
        dirs = [
            f"{self.yolo_dir}/train/images",
            f"{self.yolo_dir}/train/labels", 
            f"{self.yolo_dir}/val/images",
            f"{self.yolo_dir}/val/labels",
            f"{self.yolo_dir}/test/images",
            f"{self.yolo_dir}/test/labels"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created directory structure in {self.yolo_dir}")
    
    def create_distinct_quiet_sun(self, size=(640, 640)):
        """Create quiet sun images with smooth, uniform patterns"""
        image = np.zeros(size, dtype=np.uint8)
        
        # Create smooth background gradient
        center_x, center_y = size[1]//2, size[0]//2
        y, x = np.ogrid[:size[0], :size[1]]
        
        # Radial gradient from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Smooth intensity falloff
        base_intensity = 200 - (distance / max_distance) * 50
        
        # Add very subtle texture
        noise = np.random.normal(0, 5, size)
        
        # Combine
        image = np.clip(base_intensity + noise, 50, 200).astype(np.uint8)
        
        # Add limb darkening effect
        limb_mask = distance < (min(size) * 0.45)
        image[~limb_mask] = image[~limb_mask] * 0.3
        
        return image
    
    def create_distinct_storm(self, size=(640, 640)):
        """Create storm images with bright, active regions"""
        # Start with quiet sun base
        image = self.create_distinct_quiet_sun(size)
        
        # Add multiple bright active regions
        num_regions = random.randint(2, 5)
        
        for _ in range(num_regions):
            # Random position for active region
            center_x = random.randint(size[1]//4, 3*size[1]//4)
            center_y = random.randint(size[0]//4, 3*size[0]//4)
            
            # Create bright region
            y, x = np.ogrid[:size[0], :size[1]]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Variable size active regions
            region_size = random.randint(30, 80)
            intensity_boost = random.randint(80, 150)
            
            # Gaussian-like bright region
            region_mask = distance < region_size
            falloff = np.exp(-(distance**2) / (2 * (region_size/3)**2))
            
            image = image.astype(np.float32)
            image[region_mask] += intensity_boost * falloff[region_mask]
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def create_distinct_flare(self, size=(640, 640)):
        """Create flare images with very bright, concentrated regions"""
        # Start with quiet sun
        image = self.create_distinct_quiet_sun(size)
        
        # Add 1-2 very bright flare regions
        num_flares = random.randint(1, 2)
        
        for _ in range(num_flares):
            center_x = random.randint(size[1]//3, 2*size[1]//3)
            center_y = random.randint(size[0]//3, 2*size[0]//3)
            
            y, x = np.ogrid[:size[0], :size[1]]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Very bright, small regions
            flare_size = random.randint(15, 40)
            intensity_boost = random.randint(150, 255)
            
            # Sharp, bright peak
            flare_mask = distance < flare_size
            falloff = np.exp(-(distance**2) / (2 * (flare_size/4)**2))
            
            image = image.astype(np.float32)
            image[flare_mask] += intensity_boost * falloff[flare_mask]
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def create_distinct_cme(self, size=(640, 640)):
        """Create CME images with large-scale disturbances"""
        # Start with storm base
        image = self.create_distinct_storm(size)
        
        # Add large-scale disturbance patterns
        center_x, center_y = size[1]//2, size[0]//2
        
        # Create spiral or ejection pattern
        y, x = np.ogrid[:size[0], :size[1]]
        
        # Radial pattern simulating ejection
        angle = np.arctan2(y - center_y, x - center_x)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create disturbance pattern
        pattern = np.sin(angle * 3 + distance * 0.02) * 50
        
        # Apply to outer regions
        outer_mask = distance > min(size) * 0.2
        
        image = image.astype(np.float32)
        image[outer_mask] += pattern[outer_mask]
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def create_yolo_label(self, class_id, image_size=(640, 640)):
        """Create YOLO format label (full image classification)"""
        # For classification, we use the full image as bounding box
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0
        
        return f"{class_id} {x_center} {y_center} {width} {height}\n"
    
    def generate_enhanced_dataset(self, samples_per_class=100):
        """Generate the enhanced dataset with distinct visual characteristics"""
        print(f"🚀 Generating Enhanced Distinct Solar Dataset")
        print(f"Creating {samples_per_class} samples per class")
        
        self.create_directory_structure()
        
        generators = {
            'quiet_sun': self.create_distinct_quiet_sun,
            'active_storm': self.create_distinct_storm,
            'intense_flare': self.create_distinct_flare,
            'cme_activity': self.create_distinct_cme
        }
        
        # Split ratios
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        
        for class_name, generator_func in generators.items():
            class_id = self.classes[class_name]
            print(f"\n📸 Generating {class_name} images...")
            
            for i in range(samples_per_class):
                # Generate image
                image = generator_func()
                
                # Convert to RGB for YOLO
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image
                
                # Determine split
                if i < int(samples_per_class * train_ratio):
                    split = "train"
                elif i < int(samples_per_class * (train_ratio + val_ratio)):
                    split = "val"
                else:
                    split = "test"
                
                # Save image
                image_filename = f"{split}_{i:06d}_{class_name}.jpg"
                image_path = f"{self.yolo_dir}/{split}/images/{image_filename}"
                cv2.imwrite(image_path, image_rgb)
                
                # Save label
                label_filename = f"{split}_{i:06d}_{class_name}.txt"
                label_path = f"{self.yolo_dir}/{split}/labels/{label_filename}"
                
                with open(label_path, 'w') as f:
                    f.write(self.create_yolo_label(class_id))
                
                if i % 20 == 0:
                    print(f"  Generated {i+1}/{samples_per_class} {class_name} images")
        
        # Create dataset.yaml for YOLO
        dataset_config = {
            'path': os.path.abspath(self.yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': list(self.classes.keys())
        }
        
        with open(f"{self.yolo_dir}/dataset.yaml", 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        # Save metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'samples_per_class': samples_per_class,
            'classes': self.classes,
            'total_samples': samples_per_class * len(self.classes),
            'description': 'Enhanced dataset with visually distinct solar phenomena'
        }
        
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📁 Location: {self.output_dir}")
        print(f"📊 Total samples: {samples_per_class * len(self.classes)}")
        print(f"🏷️ Classes: {list(self.classes.keys())}")
    
    def create_sample_visualization(self):
        """Create sample images for visualization"""
        print(f"\n🎨 Creating sample visualization...")
        
        sample_dir = f"{self.output_dir}/samples"
        Path(sample_dir).mkdir(parents=True, exist_ok=True)
        
        generators = {
            'quiet_sun': self.create_distinct_quiet_sun,
            'active_storm': self.create_distinct_storm,
            'intense_flare': self.create_distinct_flare,
            'cme_activity': self.create_distinct_cme
        }
        
        for class_name, generator_func in generators.items():
            sample_image = generator_func()
            sample_path = f"{sample_dir}/sample_{class_name}.jpg"
            cv2.imwrite(sample_path, sample_image)
            print(f"  ✓ {sample_path}")

def main():
    """Main execution function"""
    print("🌞 Enhanced Solar Dataset Creator")
    print("=" * 40)
    
    creator = EnhancedSolarDatasetCreator()
    
    # Create sample visualizations first
    creator.create_sample_visualization()
    
    # Generate the full dataset
    creator.generate_enhanced_dataset(samples_per_class=150)
    
    print(f"\n🎯 Next steps:")
    print("1. Review sample images in data/enhanced_distinct_dataset/samples/")
    print("2. Train YOLO model on this new dataset")
    print("3. Test improved storm detection accuracy")

if __name__ == "__main__":
    main()