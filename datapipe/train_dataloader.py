"""
Training Data Loader
Used to load image datasets and apply RealESRGAN degradation pipeline to generate training data pairs
"""

import os
import sys

# 确保项目根目录在 sys.path 中，使 datapipe 包可被导入
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List
import cv2

from datapipe.realesrgan_degradation import RealESRGANDegradation


class RealESRGANTrainDataset(Dataset):
    """
    RealESRGAN Training Dataset
    Loads image folders, randomly crops to specified size, and applies degradation pipeline to generate LR-HR image pairs
    """
    
    def __init__(
        self,
        data_dir: str,
        config_path: str,
        gt_size: int = 512,
        use_hflip: bool = True,
        use_rot: bool = False,
        image_extensions: List[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Image data directory path
            config_path: RealESRGAN degradation configuration file path
            gt_size: Ground truth image size (after cropping)
            use_hflip: Whether to use horizontal flip augmentation
            use_rot: Whether to use rotation augmentation
            image_extensions: List of supported image extensions
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.gt_size = gt_size
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # Default supported image formats
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        
        # Recursively load all image paths (search directory and all its subdirectories)
        self.image_paths = []
        for ext in image_extensions:
            # Use ** pattern to recursively search all subdirectories
            self.image_paths.extend(glob.glob(str(self.data_dir / f'**/*{ext}'), recursive=True))
            self.image_paths.extend(glob.glob(str(self.data_dir / f'**/*{ext.upper()}'), recursive=True))
        
        # Remove duplicates and sort for reproducibility
        self.image_paths = sorted(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in directory {data_dir} and its subdirectories")
        
        print(f"Dataset initialization complete: Found {len(self.image_paths)} images in {data_dir} and its subdirectories")
        
        # Initialize degradation pipeline
        self.degrader = RealESRGANDegradation(config_path)
        
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_paths)
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load image
        
        Args:
            img_path: Image path
        
        Returns:
            RGB format numpy array, value range [0, 255]
        """
        # Use cv2 to read image (BGR format)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _random_crop_or_pad(self, img: np.ndarray, crop_size: int) -> np.ndarray:
        """
        Randomly crop or pad image to specified size
        
        Args:
            img: Input image [H, W, C]
            crop_size: Target size
        
        Returns:
            Cropped/padded image [crop_size, crop_size, C]
        """
        h, w = img.shape[:2]
        
        # If image is smaller than target size, pad first
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            
            # Use reflection padding
            img = cv2.copyMakeBorder(
                img,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_REFLECT_101
            )
            h, w = img.shape[:2]
        
        # Random crop
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        img_crop = img[top:top + crop_size, left:left + crop_size, :]
        return img_crop
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        Data augmentation
        
        Args:
            img: Input image [H, W, C]
        
        Returns:
            Augmented image
        """
        # Horizontal flip
        if self.use_hflip and random.random() < 0.5:
            img = cv2.flip(img, 1)
        
        # Rotation (multiples of 90 degrees)
        if self.use_rot:
            rot_times = random.randint(0, 3)
            if rot_times > 0:
                img = np.rot90(img, rot_times)
        
        return img
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        If image loading or degradation fails, randomly retry another sample.
        """
        for _ in range(10):
            try:
                img_path = self.image_paths[index]
                img = self._load_image(img_path)
                
                img_gt = self._random_crop_or_pad(img, self.gt_size)
                img_gt = self._augment(img_gt)
                
                img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1).copy()).float() / 255.0
                img_gt_batch = img_gt.unsqueeze(0)
                
                result = self.degrader.degrade(img_gt_batch)
                
                img_lq = result['lq'].squeeze(0).detach()
                img_gt_out = result['gt'].squeeze(0).detach()
                
                return {
                    'lq': img_lq,
                    'gt': img_gt_out,
                    'lq_path': img_path,
                }
            except Exception as e:
                print(f"[DataLoader] Error loading index {index}: {e}, retrying...")
                index = random.randint(0, len(self.image_paths) - 1)
        
        raise RuntimeError(f"Failed to load any valid image after 10 retries")


def create_train_dataloader(
    data_dir: str,
    config_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    gt_size: int = 512,
    use_hflip: bool = True,
    use_rot: bool = False,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create training data loader
    
    Args:
        data_dir: Image data directory path
        config_path: RealESRGAN degradation configuration file path
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        gt_size: Ground truth image size
        use_hflip: Whether to use horizontal flip
        use_rot: Whether to use rotation
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin data in memory (accelerate GPU transfer)
    
    Returns:
        DataLoader object
    """
    dataset = RealESRGANTrainDataset(
        data_dir=data_dir,
        config_path=config_path,
        gt_size=gt_size,
        use_hflip=use_hflip,
        use_rot=use_rot
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop the last incomplete batch
    )
    
    return dataloader


def test_dataloader():
    """Test data loader"""
    import matplotlib.pyplot as plt
    
    # Configuration paths
    data_dir = Path(__file__).parent.parent / 'testdata'
    config_path = Path(__file__).parent.parent / 'configs' / 'realesrgan_degradation.yaml'
    
    print(f"Data directory: {data_dir}")
    print(f"Configuration file: {config_path}")
    
    # Create data loader
    dataloader = create_train_dataloader(
        data_dir=str(data_dir),
        config_path=str(config_path),
        batch_size=4,
        num_workers=2,  # Use 0 during testing to avoid multiprocessing issues
        gt_size=512,
        use_hflip=True,
        use_rot=False,
        shuffle=True
    )
    
    print(f"Data loader created successfully, total {len(dataloader)} batches")
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch information:")
    print(f"  LQ shape: {batch['lq'].shape}")
    print(f"  GT shape: {batch['gt'].shape}")
    print(f"  LQ value range: [{batch['lq'].min():.3f}, {batch['lq'].max():.3f}]")
    print(f"  GT value range: [{batch['gt'].min():.3f}, {batch['gt'].max():.3f}]")
    
    # Visualize first sample
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for i in range(2):
        # Convert to numpy array [C, H, W] -> [H, W, C]
        img_gt = batch['gt'][i].permute(1, 2, 0).cpu().numpy()
        img_lq = batch['lq'][i].permute(1, 2, 0).cpu().numpy()
        
        # Display GT
        axes[i, 0].imshow(img_gt)
        axes[i, 0].set_title(f'Sample {i+1} - GT (HR)', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Display LQ
        axes[i, 1].imshow(img_lq)
        axes[i, 1].set_title(f'Sample {i+1} - LQ (Degraded)', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test iterating multiple batches
    print("\nTesting iteration of 3 batches...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        print(f"Batch {i+1}: LQ shape={batch['lq'].shape}, GT shape={batch['gt'].shape}")
    
    print("\nData loader test completed!")


if __name__ == '__main__':
    test_dataloader()
