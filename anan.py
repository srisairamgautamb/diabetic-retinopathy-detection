# Fragment 1: Imports and Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from prefetch_generator import BackgroundGenerator
warnings.filterwarnings('ignore')

# Fragment 2: Custom DataLoader for Background Loading
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Fragment 3: Advanced Image Preprocessing
class AdvancedPreprocessing:
    def __init__(self, image_size=512):
        self.image_size = image_size
        self.cache = {}
        
    @torch.no_grad()
    def preprocess_image(self, image_path):
        if image_path in self.cache:
            return self.cache[image_path]
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Optimize mask creation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Cache the result
        self.cache[image_path] = img
        return img

# Fragment 4: Dataset Class
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, preprocessing=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id_code']
        img_path = f"{self.img_dir}/{img_name}.png"
        
        # Preprocess image
        image = self.preprocessing.preprocess_image(img_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        if self.is_test:
            return image
        else:
            label = self.df.iloc[idx]['diagnosis']
            return image, label

# Fragment 5: Model Architecture
class EfficientNetWithMixup(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        in_features = self.model.classifier.in_features
        
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# Fragment 6: Mixup Implementation
def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Fragment 7: Training Functions
def train_fold(fold, model, train_loader, valid_loader, device, criterion, optimizer, scheduler, num_epochs=15):
    scaler = GradScaler()
    best_score = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup to training data
            inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2, device=device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs_mixed)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        valid_preds = []
        valid_targets = []
        valid_loss = 0
        
        with torch.no_grad(), autocast():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_targets.extend(targets.cpu().numpy())
        
        valid_score = cohen_kappa_score(valid_targets, valid_preds, weights='quadratic')
        
        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
            
        scheduler.step(valid_score)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Valid Loss: {valid_loss/len(valid_loader):.4f}')
        print(f'Valid Kappa Score: {valid_score:.4f}')
        print(f'Best Kappa Score: {best_score:.4f}\n')
    
    return best_score

# Fragment 8: Testing and Prediction Functions
def prepare_test_data(test_df, preprocessing, config, transforms):
    """Prepare test dataset and dataloader"""
    test_dataset = RetinopathyDataset(
        test_df,
        f"{config['DATA_PATH']}/test_images",
        transform=transforms,
        preprocessing=preprocessing,
        is_test=True
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=config['BATCH_SIZE'] * 2,
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY']
    )
    
    return test_loader

def make_predictions(model, test_loader, device):
    """Make predictions on test data using trained model"""
    model.eval()
    predictions = []
    
    with torch.no_grad(), autocast():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions

def ensemble_predictions(config, test_df, preprocessing, transforms, device):
    """Ensemble predictions from all trained model folds"""
    all_predictions = []
    
    for fold in range(1, config['N_FOLDS'] + 1):
        try:
            # Load model for current fold
            model = EfficientNetWithMixup(num_classes=5).to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            
            model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
            
            # Prepare test data
            test_loader = prepare_test_data(test_df, preprocessing, config, transforms)
            
            # Make predictions
            fold_predictions = make_predictions(model, test_loader, device)
            all_predictions.append(fold_predictions)
            
            print(f"Completed predictions for fold {fold}")
            
        except Exception as e:
            print(f"Error making predictions for fold {fold}: {str(e)}")
            continue
    
    # Average predictions from all folds
    if all_predictions:
        final_predictions = np.mean(all_predictions, axis=0)
        return np.round(final_predictions).astype(int)
    else:
        raise ValueError("No successful predictions made")

def generate_submission(config):
    """Generate submission file with test predictions"""
    try:
        # Read test data
        test_df = pd.read_csv(f"{config['DATA_PATH']}/test.csv")
        
        # Initialize preprocessing and transforms
        preprocessing = AdvancedPreprocessing(image_size=config['IMAGE_SIZE'])
        transforms = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get ensemble predictions
        predictions = ensemble_predictions(config, test_df, preprocessing, transforms, device)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id_code': test_df['id_code'],
            'diagnosis': predictions
        })
        
        # Save submission file
        submission_df.to_csv('submission.csv', index=False)
        print("Submission file generated successfully!")
        
    except Exception as e:
        print(f"Error generating submission: {str(e)}")
        import traceback
        traceback.print_exc()

# Fragment 9: Main Function
def main():
    config = {
        'SEED': 42,
        'IMAGE_SIZE': 384,
        'BATCH_SIZE': 32,
        'NUM_EPOCHS': 15,
        'N_FOLDS': 5,
        'LEARNING_RATE': 2e-4,
        'WEIGHT_DECAY': 1e-5,
        'NUM_WORKERS': 4,
        'PIN_MEMORY': True,
        'DATA_PATH': '/kaggle/input/aptos2019-blindness-detection',
    }
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config['SEED'])
    np.random.seed(config['SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['SEED'])
    
    try:
        train_df = pd.read_csv(f"{config['DATA_PATH']}/train.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find train.csv at {config['DATA_PATH']}")
        return
    
    preprocessing = AdvancedPreprocessing(image_size=config['IMAGE_SIZE'])
    
    # Training transforms
    transforms = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    
    skf = StratifiedKFold(n_splits=config['N_FOLDS'], shuffle=True, random_state=config['SEED'])
    scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df, train_df['diagnosis']), 1):
        print(f'\nTraining Fold {fold}/{config["N_FOLDS"]}')
        
        try:
            train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
            valid_fold_df = train_df.iloc[valid_idx].reset_index(drop=True)
            
            train_dataset = RetinopathyDataset(
                train_fold_df,
                f"{config['DATA_PATH']}/train_images",
                transform=transforms,
                preprocessing=preprocessing
            )
            
            valid_dataset = RetinopathyDataset(
                valid_fold_df,
                f"{config['DATA_PATH']}/train_images",
                transform=transforms,
                preprocessing=preprocessing
            )
            
            train_loader = DataLoaderX(
                train_dataset,
                batch_size=config['BATCH_SIZE'],
                shuffle=True,
                num_workers=config['NUM_WORKERS'],
                pin_memory=config['PIN_MEMORY']
            )
            
            valid_loader = DataLoaderX(
                valid_dataset,
                batch_size=config['BATCH_SIZE'] * 2,
                shuffle=False,
                num_workers=config['NUM_WORKERS'],
                pin_memory=config['PIN_MEMORY']
            )
            
            model = EfficientNetWithMixup(num_classes=5).to(device)
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                model = nn.DataParallel(model)
                
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['LEARNING_RATE'],
                weight_decay=config['WEIGHT_DECAY']
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2, verbose=True
            )
            
            score = train_fold(
                fold, model, train_loader, valid_loader,
                device, criterion, optimizer, scheduler,
                config['NUM_EPOCHS']
            )
            scores.append(score)
            
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if scores:
        print("\nCross-validation scores:", scores)
        print(f"Mean score: {np.mean(scores):.4f}")
        print(f"Std score: {np.std(scores):.4f}")
    
    # Generate submission file
    try:
        generate_submission(config)
    except Exception as e:
        print(f"Error generating submission: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
