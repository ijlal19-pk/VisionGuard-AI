import torch
import sys
import os

# Add workspace directory to python path to resolve imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import ResNet50_CBAM_MultiTask, EfficientNetB0_CBAM_MultiTask

def test_models():
    print("==================================================")
    print("STARTING MODEL PIPELINE VERIFICATION")
    print("==================================================")
    
    # 1. Create a dummy input batch: size 1, 3 color channels, 224x224 pixels
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Created dummy input tensor with shape: {dummy_input.shape}")
    
    # 2. Test the ResNet-50 Teacher Model
    print("\nTesting Teacher Model (ResNet-50 + CBAM + Decoder)...")
    try:
        # Load without pretrained weights to run quickly without internet downloads
        teacher_model = ResNet50_CBAM_MultiTask(pretrained=False)
        teacher_model.eval()
        
        with torch.no_grad():
            class_out, seg_out = teacher_model(dummy_input)
            
        print("[OK] Teacher Model Forward Pass Successful!")
        print(f" -> Classification Logits shape: {class_out.shape} (Expected: [1, 2])")
        print(f" -> Segmentation Masks shape:    {seg_out.shape} (Expected: [1, 2, 224, 224])")
        
        # Verify classes and dimensions
        assert class_out.shape == (1, 2), f"Incorrect classification shape: {class_out.shape}"
        assert seg_out.shape == (1, 2, 224, 224), f"Incorrect segmentation shape: {seg_out.shape}"
        print("[OK] Teacher Output Shapes Verified!")
        
    except Exception as e:
        print(f"[ERROR] Teacher Model Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    # 3. Test the EfficientNet-B0 Student Model
    print("\nTesting Student Model (EfficientNet-B0 + Light CBAM + Decoder)...")
    try:
        # Load without pretrained weights to run quickly without internet downloads
        student_model = EfficientNetB0_CBAM_MultiTask(pretrained=False)
        student_model.eval()
        
        with torch.no_grad():
            class_out, seg_out = student_model(dummy_input)
            
        print("[OK] Student Model Forward Pass Successful!")
        print(f" -> Classification Logits shape: {class_out.shape} (Expected: [1, 2])")
        print(f" -> Segmentation Masks shape:    {seg_out.shape} (Expected: [1, 2, 224, 224])")
        
        # Verify classes and dimensions
        assert class_out.shape == (1, 2), f"Incorrect classification shape: {class_out.shape}"
        assert seg_out.shape == (1, 2, 224, 224), f"Incorrect segmentation shape: {seg_out.shape}"
        print("[OK] Student Output Shapes Verified!")
        
    except Exception as e:
        print(f"[ERROR] Student Model Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n==================================================")
    print("ALL MODELS VERIFIED SUCCESSFULLY!")
    print("==================================================")
    return True


if __name__ == "__main__":
    test_models()
