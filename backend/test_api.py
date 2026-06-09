import os
import io
import sys
import numpy as np
from PIL import Image

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import TestClient and the FastAPI app
try:
    from fastapi.testclient import TestClient
    from backend.main import app
except ImportError:
    print("[ERROR] FastAPI or TestClient not installed. Run 'pip install fastapi uvicorn' first.")
    sys.exit(1)

def run_api_tests():
    print("==================================================")
    print("STARTING INTEGRATED API PIPELINE VERIFICATION")
    print("==================================================")
    
    # Use context manager to trigger FastAPI startup event (model initialization)
    with TestClient(app) as client:
        print("FastAPI TestClient initialized with context manager.")
        
        # 2. Create a mock RGB fundus image (224x224)
        mock_img_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        mock_pil = Image.fromarray(mock_img_arr)
        
        img_byte_arr = io.BytesIO()
        mock_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        files = {
            "file": ("mock_fundus.jpg", img_byte_arr, "image/jpeg")
        }
        
        # 3. Test endpoint /api/screen?model_type=student
        print("\nTesting POST /api/screen with student model...")
        try:
            response = client.post("/api/screen?model_type=student", files=files)
            print(f" -> Response Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"API error response: {response.text}"
            res_json = response.json()
            
            print("[OK] Response keys returned:")
            for key, val in res_json.items():
                if isinstance(val, str) and len(val) > 100:
                    print(f"   - {key}: [Base64 Encoded Image, length={len(val)}]")
                else:
                    print(f"   - {key}: {val}")
                    
            # Verification of schema
            required_keys = ["model_type", "diagnosis", "confidence", "cup_to_disc_ratio", "recommendation", "gradcam_heatmap", "segmentation_overlay"]
            for key in required_keys:
                assert key in res_json, f"Missing required response key: {key}"
                
            print("[OK] Student Model API Response Schema Verified!")
            
        except Exception as e:
            print(f"[ERROR] Student Model API Test Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        # 4. Reset byte array and test teacher model
        img_byte_arr.seek(0)
        files = {
            "file": ("mock_fundus.jpg", img_byte_arr, "image/jpeg")
        }
        
        print("\nTesting POST /api/screen with teacher model...")
        try:
            response = client.post("/api/screen?model_type=teacher", files=files)
            print(f" -> Response Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"API error response: {response.text}"
            res_json = response.json()
            
            # Verification of schema
            for key in required_keys:
                assert key in res_json, f"Missing required response key: {key}"
                
            print("[OK] Teacher Model API Response Schema Verified!")
            
        except Exception as e:
            print(f"[ERROR] Teacher Model API Test Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        # 5. Test metrics endpoint
        print("\nTesting GET /api/metrics...")
        try:
            response = client.get("/api/metrics")
            print(f" -> Response Status Code: {response.status_code}")
            
            assert response.status_code == 200, f"API error response: {response.text}"
            metrics = response.json()
            
            assert "teacher" in metrics and "student" in metrics, "Metrics should contain teacher and student records"
            print("[OK] Performance Metrics API Verified!")
            
        except Exception as e:
            print(f"[ERROR] Metrics API Test Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    print("\n==================================================")
    print("INTEGRATED API PIPELINE VERIFIED SUCCESSFULLY!")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_api_tests()
