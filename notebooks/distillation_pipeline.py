import os
import time
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

# ==========================================
# 0. MODEL DEFINITIONS (Copied for Colab Portability)
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP structure
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class ResNet50_CBAM_MultiTask(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_CBAM_MultiTask, self).__init__()
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet50()
            
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.cbam = CBAM(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 2)
        
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4)
        self.conv_dec4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, x):
        s = self.stem(x)
        l1 = self.layer1(s)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        
        refined = self.cbam(l4)
        
        pool = self.avgpool(refined)
        flat = torch.flatten(pool, 1)
        class_out = self.classifier(flat)
        
        d1 = self.up1(refined)
        d1 = torch.cat([d1, l3], dim=1)
        d1 = self.conv_dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, l2], dim=1)
        d2 = self.conv_dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, l1], dim=1)
        d3 = self.conv_dec3(d3)
        
        d4 = self.up4(d3)
        seg_out = self.conv_dec4(d4)
        
        return class_out, seg_out


class EfficientNetB0_CBAM_MultiTask(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0_CBAM_MultiTask, self).__init__()
        if pretrained:
            effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            effnet = models.efficientnet_b0()
            
        self.features = effnet.features
        self.cbam = CBAM(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, 2)
        
        self.up1 = nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2)
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(256 + 112, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(64 + 40, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(32 + 24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4)
        self.conv_dec4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1)
        )

    def forward(self, x):
        x0_1 = self.features[0](x)
        x0_1 = self.features[1](x0_1)
        x2 = self.features[2](x0_1)
        x3 = self.features[3](x2)
        x4 = self.features[4](x3)
        x5 = self.features[5](x4)
        x6 = self.features[6](x5)
        x7 = self.features[7](x6)
        x8 = self.features[8](x7)
        
        refined = self.cbam(x8)
        
        pool = self.avgpool(refined)
        flat = torch.flatten(pool, 1)
        class_out = self.classifier(flat)
        
        d1 = self.up1(refined)
        d1 = torch.cat([d1, x5], dim=1)
        d1 = self.conv_dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.conv_dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv_dec3(d3)
        
        d4 = self.up4(d3)
        seg_out = self.conv_dec4(d4)
        
        return class_out, seg_out

# ==========================================
# 1. LOSS FUNCTIONS & METRICS
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        # y_pred: [B, C, H, W] (after sigmoid activation)
        # y_true: [B, C, H, W] (binary mask)
        y_pred = torch.sigmoid(y_pred)
        
        intersection = torch.sum(y_pred * y_true, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true, dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - torch.mean(dice)

class JointLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(JointLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss_dice = DiceLoss()
        self.segmentation_loss_bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha # Classification loss weight
        self.beta = beta   # Segmentation loss weight

    def forward(self, pred_class, target_class, pred_seg, target_seg):
        loss_class = self.classification_loss(pred_class, target_class)
        loss_seg_dice = self.segmentation_loss_dice(pred_seg, target_seg)
        loss_seg_bce = self.segmentation_loss_bce(pred_seg, target_seg)
        loss_seg = loss_seg_dice + loss_seg_bce
        
        total_loss = self.alpha * loss_class + self.beta * loss_seg
        return total_loss, loss_class, loss_seg

def distillation_loss(student_logits, teacher_logits, temp=4.0, alpha_kd=0.7):
    # KL Divergence for soft labels (classification distillation)
    p_teacher = F.softmax(teacher_logits / temp, dim=1)
    log_p_student = F.log_softmax(student_logits / temp, dim=1)
    kd_loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temp ** 2)
    return kd_loss

# ==========================================
# 2. DATASET CLASS FOR MULTI-TASK DATA
# ==========================================

class GlaucomaMultiTaskDataset(Dataset):
    def __init__(self, image_paths, labels, mask_paths=None, transform=None, mask_transform=None):
        """
        image_paths: list of paths to fundus images
        labels: list of binary labels (0 = Normal, 1 = Glaucoma)
        mask_paths: list of paths to segmentation masks (None for test datasets without ground-truth masks)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        # Apply image transformations
        if self.transform:
            image = self.transform(image)
            
        # Load and process masks (only if ground truth mask is available)
        if self.mask_paths and self.mask_paths[idx] is not None:
            mask_path = self.mask_paths[idx]
            # Mask is typically a grayscale image
            mask_img = Image.open(mask_path).convert("L")
            
            # Preprocess masks:
            # Optic disc usually has one intensity value, and optic cup another.
            # Convert to numpy array to split channels.
            mask_arr = np.array(mask_img)
            
            # Let's assume standard values (e.g. REFUGE: disc <= 128, cup <= 0, background 255)
            # We map: Disc mask = 1 where disc is present, Cup mask = 1 where cup is present.
            # You can customize thresholds based on the dataset mapping.
            disc_mask = (mask_arr < 255).astype(np.float32) # Anything not white is disc
            cup_mask = (mask_arr < 128).astype(np.float32)  # Darker regions are cup
            
            # Combine into 2-channel array: [2, H, W]
            combined_mask = np.stack([disc_mask, cup_mask], axis=0)
            combined_mask_tensor = torch.from_numpy(combined_mask)
            
            # If resizing masks is needed (e.g. to 224x224)
            if self.mask_transform:
                # Resize requires [C, H, W] tensor
                combined_mask_tensor = self.mask_transform(combined_mask_tensor)
        else:
            # Return dummy mask tensor [2, 224, 224] for datasets without ground-truth segmentations (e.g. ACRIMA)
            combined_mask_tensor = torch.zeros(2, 224, 224, dtype=torch.float32)
            
        return image, label, combined_mask_tensor

# ==========================================
# 3. TRAINING & DISTILLATION LOOPS
# ==========================================

def train_teacher(model, train_loader, val_loader, epochs, device, save_path):
    print("------------------------------------------")
    print("TRAINING TEACHER MODEL (ResNet-50 + CBAM)")
    print("------------------------------------------")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = JointLoss(alpha=1.0, beta=1.0)
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_class = 0
        total_samples = 0
        
        for images, labels, masks in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_out, seg_out = model(images)
            
            # Compute loss
            loss, loss_cls, loss_seg = criterion(class_out, labels, seg_out, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(class_out, 1)
            correct_class += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_class / total_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                class_out, seg_out = model(images)
                loss, _, _ = criterion(class_out, labels, seg_out, masks)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(class_out, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += images.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(epoch_val_acc)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f" -> Saved Best Teacher Model weights to {save_path}")
            
    return history


def train_student_with_distillation(student, teacher, train_loader, val_loader, epochs, device, save_path, teacher_weights_path):
    print("\n------------------------------------------")
    print("TRAINING STUDENT MODEL WITH DISTILLATION")
    print("------------------------------------------")
    
    # Load trained teacher weights
    teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    teacher = teacher.to(device)
    teacher.eval() # Teacher is always in evaluation mode
    
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=2e-4, weight_decay=1e-5)
    
    # Ground truth loss
    criterion_gt = JointLoss(alpha=0.5, beta=1.0)
    # Dice loss for matching student mask to teacher mask (segmentation distillation)
    criterion_seg_kd = DiceLoss()
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        correct_class = 0
        total_samples = 0
        
        for images, labels, masks in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (student)
            s_class_out, s_seg_out = student(images)
            
            # Forward pass (teacher - no gradients)
            with torch.no_grad():
                t_class_out, t_seg_out = teacher(images)
                
            # 1. Ground truth multi-task loss
            loss_gt, loss_cls_gt, loss_seg_gt = criterion_gt(s_class_out, labels, s_seg_out, masks)
            
            # 2. Knowledge distillation loss (Classification: KL Divergence)
            loss_kd_cls = distillation_loss(s_class_out, t_class_out, temp=4.0, alpha_kd=0.7)
            
            # 3. Knowledge distillation loss (Segmentation: Student Mimics Teacher Mask)
            loss_kd_seg = criterion_seg_kd(s_seg_out, torch.sigmoid(t_seg_out))
            
            # Joint Distillation Loss
            total_loss = loss_gt + 0.5 * loss_kd_cls + 0.5 * loss_kd_seg
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * images.size(0)
            _, preds = torch.max(s_class_out, 1)
            correct_class += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_class / total_samples
        
        # Validation
        student.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                s_class_out, s_seg_out = student(images)
                loss, _, _ = criterion_gt(s_class_out, labels, s_seg_out, masks)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(s_class_out, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += images.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | Distill Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(epoch_val_acc)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(student.state_dict(), save_path)
            print(f" -> Saved Best Student Model weights to {save_path}")
            
    return history

# ==========================================
# 4. CROSS-DATASET EVALUATION SCRIPT
# ==========================================

def run_cross_dataset_eval(model_weights_path, model_type, test_loaders_dict, device):
    """
    model_weights_path: Path to .pth file
    model_type: 'teacher' or 'student'
    test_loaders_dict: dictionary of {dataset_name: dataloader}
    """
    print(f"\n--- Running Cross-Dataset Evaluation for: {model_type.upper()} ---")
    if model_type == 'teacher':
        model = ResNet50_CBAM_MultiTask(pretrained=False)
    else:
        model = EfficientNetB0_CBAM_MultiTask(pretrained=False)
        
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for name, loader in test_loaders_dict.items():
            print(f"Evaluating on {name}...")
            correct = 0
            total = 0
            
            for images, labels, _ in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                class_out, _ = model(images)
                _, preds = torch.max(class_out, 1)
                
                correct += torch.sum(preds == labels.data).item()
                total += images.size(0)
                
            accuracy = correct / total if total > 0 else 0
            print(f" -> Accuracy on {name}: {accuracy*100:.2f}%")
            results[name] = accuracy
            
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Paths (Pointing to your Google Drive)
    base_data_dir = "/content/drive/MyDrive/Glaucoma_Project/data"
    train_dir = os.path.join(base_data_dir, "train")
    val_dir = os.path.join(base_data_dir, "val")
    
    if not os.path.exists(train_dir):
        print(f"ERROR: Cannot find training data at {train_dir}")
        print("Please ensure you uploaded the 'data' folder to Colab Google Drive correctly.")
        sys.exit(1)

    # 2. Image Transformations (Resizing, Augmentation, Normalization)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Parse Folder Structure (train/normal vs train/glaucoma)
    def parse_folder(directory):
        img_paths, labels = [], []
        for class_name, label in [("normal", 0), ("glaucoma", 1)]:
            # Check lowercase and uppercase folder names
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                class_dir = os.path.join(directory, class_name.capitalize())
            
            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                        img_paths.append(os.path.join(class_dir, fname))
                        labels.append(label)
        return img_paths, labels

    train_imgs, train_labels = parse_folder(train_dir)
    val_imgs, val_labels = parse_folder(val_dir)
    
    print(f"Found {len(train_imgs)} Training Images")
    print(f"Found {len(val_imgs)} Validation Images")
    
    if len(train_imgs) == 0:
        print("No images found! Halting.")
        sys.exit(1)
        
    # 4. Create Datasets and Loaders
    # We pass mask_paths=None. The GlaucomaMultiTaskDataset automatically 
    # creates dummy masks so the multi-task network doesn't crash during training!
    train_dataset = GlaucomaMultiTaskDataset(train_imgs, train_labels, mask_paths=None, transform=train_transform)
    val_dataset = GlaucomaMultiTaskDataset(val_imgs, val_labels, mask_paths=None, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 5. Instantiate Models with ImageNet pre-trained backbones
    teacher = ResNet50_CBAM_MultiTask(pretrained=True)
    student = EfficientNetB0_CBAM_MultiTask(pretrained=True)
    
    # Google Drive Save Paths
    teacher_save_path = "/content/drive/MyDrive/Glaucoma_Project/teacher_model.pth"
    student_save_path = "/content/drive/MyDrive/Glaucoma_Project/student_model.pth"
    
    # 6. Run Full Distillation Pipeline
    print("\n--- STARTING STAGE 1: TEACHER TRAINING (ResNet-50) ---")
    train_teacher(teacher, train_loader, val_loader, epochs=15, device=device, save_path=teacher_save_path)
    
    print("\n--- STARTING STAGE 2: KNOWLEDGE DISTILLATION (Student EfficientNet-B0) ---")
    train_student_with_distillation(student, teacher, train_loader, val_loader, epochs=15, device=device, save_path=student_save_path, teacher_weights_path=teacher_save_path)
    
    print("\n[SUCCESS] Training Complete! Download the two .pth files from your Google Drive.")
