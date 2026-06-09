import os
import shutil
import random

# Configuration
RAW_DIR = r"c:\Users\ijlal\Desktop\GLUCOMA DETECTOR\data\raw"
BASE_DIR = r"c:\Users\ijlal\Desktop\GLUCOMA DETECTOR\data"
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

def split_data():
    # 1. Get all images
    all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_files:
        print(f"No images found in {RAW_DIR}. Please move your images there first!")
        return

    # 2. Separate by class based on ACRIMA naming
    # Glaucoma images have '_g_' in the filename (e.g., Im311_g_ACRIMA.jpg)
    # Normal images do NOT have '_g_' (e.g., Im001_ACRIMA.jpg)
    glaucoma_files = [f for f in all_files if '_g_' in f.lower()]
    normal_files = [f for f in all_files if '_g_' not in f.lower()]

    print(f"Found: {len(glaucoma_files)} Glaucoma images, {len(normal_files)} Normal images.")

    # 3. Process each class
    for class_name, files in [('glaucoma', glaucoma_files), ('normal', normal_files)]:
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * SPLITS['train'])
        n_val = int(n_total * SPLITS['val'])
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # Move files
        for split, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            dest_dir = os.path.join(BASE_DIR, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in split_files:
                src = os.path.join(RAW_DIR, f)
                dst = os.path.join(dest_dir, f)
                shutil.copy(src, dst)
            
            print(f"Moved {len(split_files)} images to {split}/{class_name}")

if __name__ == "__main__":
    split_data()
    print("\nData splitting complete! You are ready to train.")
