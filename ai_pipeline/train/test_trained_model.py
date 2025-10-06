#!/usr/bin/env python3
"""
Test Solar Storm Detection Model
Test the trained YOLO model on sample images.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

def test_solar_model():
    """Test the trained solar storm detection model."""
    
    # Model path
    model_path = "models/minimal_solar/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        return False
    
    print(f"🚀 Loading model from: {model_path}")
    
    try:
        # Load the trained model
        model = YOLO(model_path, verbose=False)
        print("✅ Model loaded successfully!")
        
        # Test on available solar images
        test_images_dir = Path("data/solar_images/2012/01/193")
        
        if not test_images_dir.exists():
            print(f"❌ Test images directory not found: {test_images_dir}")
            return False
        
        # Get list of test images
        image_files = list(test_images_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"❌ No test images found in: {test_images_dir}")
            return False
        
        print(f"🔍 Found {len(image_files)} test images")
        
        # Test on first few images
        for i, image_path in enumerate(image_files[:3]):
            print(f"\n📸 Testing image {i+1}: {image_path.name}")
            
            # Run inference
            results = model(str(image_path), verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    print(f"   🎯 Detected {len(boxes)} objects:")
                    
                    for j, box in enumerate(boxes):
                        # Get confidence and class
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        print(f"      Object {j+1}: {class_name} (confidence: {conf:.3f})")
                        print(f"      Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    print("   ✨ No objects detected")
            
            # Save annotated image
            annotated_img = results[0].plot()
            output_path = f"models/minimal_solar/test_result_{i+1}.jpg"
            cv2.imwrite(output_path, annotated_img)
            print(f"   💾 Saved annotated image: {output_path}")
        
        print(f"\n🎉 Model testing completed successfully!")
        print(f"📊 Model info:")
        print(f"   Classes: {model.names}")
        print(f"   Input size: {model.model.yaml.get('imgsz', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_solar_model()
    
    if success:
        print("\n✅ Solar storm detection model is working correctly!")
    else:
        print("\n❌ Model testing failed")
        sys.exit(1)