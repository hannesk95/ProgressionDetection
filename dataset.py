import os
import torch
import numpy as np
import torchio as tio
import pandas as pd
import monai
from glob import glob
from torch.utils.data import Dataset
from monai.transforms import Compose
from pathlib import Path


class BurdenkoLumiereDataset(Dataset):
    def __init__(self, split: str):
        self.split = split

        # MONAI Augmentations
        rotate = monai.transforms.RandRotate(prob=0.2, range_x=10, range_y=10, range_z=10)
        scale = monai.transforms.RandZoom(prob=0.2, min_zoom=0.7, max_zoom=1.4)
        gaussian_noise = monai.transforms.RandGaussianNoise()
        gaussian_blur = monai.transforms.RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 10.0), sigma_z=(0.5, 1.0))
        contrast = monai.transforms.RandAdjustContrast()
        intensity = monai.transforms.RandScaleIntensity(factors=(2, 10))
        histogram_shift = monai.transforms.RandHistogramShift()
        self.transforms = Compose([rotate, scale, gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift])

        if self.split == "train":
            self.data = pd.read_csv("/home/johannes/Code/ProgressionDetection/data/burdenko_lumiere_pre_post_pairs_and_labels_train.csv")
        
        elif self.split == "val":
            self.data = pd.read_csv("/home/johannes/Code/ProgressionDetection/data/burdenko_lumiere_pre_post_pairs_and_labels_val.csv")            
        
        elif self.split == "test":
            self.data = pd.read_csv("/home/johannes/Code/ProgressionDetection/data/burdenko_lumiere_pre_post_pairs_and_labels_test.csv")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):   

        patient_id = self.data.iloc[idx]["Patient-pre"]
        patient_dir = Path(self.data.iloc[idx]["Directory-pre"]).parent
        pre_dir = self.data.iloc[idx]["Directory-pre"]
        post_dir = self.data.iloc[idx]["Directory-post"]
        label = self.data.iloc[idx]["Rating-post"]
        label = 1 if label == "progression" else 0
        label = torch.tensor(label)
        roi_file = glob(os.path.join(patient_dir, "region_of_interest.nii.gz"))[0]

        if patient_id.startswith("Patient"):
            t1n_pre = glob(os.path.join(pre_dir, "*T1_processed*"))[0]
            t1n_post = glob(os.path.join(post_dir, "*T1_processed*"))[0]

            t1c_pre = glob(os.path.join(pre_dir, "*CT1_processed*"))[0]
            t1c_post = glob(os.path.join(post_dir, "*CT1_processed*"))[0]

            t2n_pre = glob(os.path.join(pre_dir, "*T2_processed*"))[0]
            t2n_post = glob(os.path.join(post_dir, "*T2_processed*"))[0]

            t2f_pre = glob(os.path.join(pre_dir, "*FLAIR_processed*"))[0]
            t2f_post = glob(os.path.join(post_dir, "*FLAIR_processed*"))[0]
        
        elif patient_id.startswith("Burdenko"):
            t1n_pre = glob(os.path.join(pre_dir, "*MR00T1_processed*"))[0]
            t1n_post = glob(os.path.join(post_dir, "*MR00T1_processed*"))[0]

            t1c_pre = glob(os.path.join(pre_dir, "*MRCET1_processed*"))[0]
            t1c_post = glob(os.path.join(post_dir, "*MRCET1_processed*"))[0]

            t2n_pre = glob(os.path.join(pre_dir, "*MR00T2_processed*"))[0]
            t2n_post = glob(os.path.join(post_dir, "*MR00T2_processed*"))[0]

            t2f_pre = glob(os.path.join(pre_dir, "*FLAIR_processed*"))[0]
            t2f_post = glob(os.path.join(post_dir, "*FLAIR_processed*"))[0]

        t1n_pre_img = self.crop_with_margin(img_path=t1n_pre, seg_path=roi_file, margin=0.2)
        t1n_post_img = self.crop_with_margin(img_path=t1n_post, seg_path=roi_file, margin=0.2)

        t1c_pre_img = self.crop_with_margin(img_path=t1c_pre, seg_path=roi_file, margin=0.2)
        t1c_post_img = self.crop_with_margin(img_path=t1c_post, seg_path=roi_file, margin=0.2)

        t2n_pre_img = self.crop_with_margin(img_path=t2n_pre, seg_path=roi_file, margin=0.2)
        t2n_post_img = self.crop_with_margin(img_path=t2n_post, seg_path=roi_file, margin=0.2)

        t2f_pre_img = self.crop_with_margin(img_path=t2f_pre, seg_path=roi_file, margin=0.2)
        t2f_post_img = self.crop_with_margin(img_path=t2f_post, seg_path=roi_file, margin=0.2)

        image = torch.concat([t1n_pre_img, t1c_pre_img, t2n_pre_img, t2f_pre_img, t1n_post_img, t1c_post_img, t2n_post_img, t2f_post_img], dim=0)        

        if self.split == "train":
            image = self.transforms(image)
            image = image.as_tensor()

        return image, label
    
    def crop_with_margin(self, img_path: str, seg_path: str, margin: float = 0.2) -> torch.Tensor:

        subject = tio.Subject(img = tio.ScalarImage(img_path),
                              seg = tio.LabelMap(seg_path))

        seg_arr = subject.seg.numpy()[0]
        seg_arr = (seg_arr > 0).astype(int)

        foreground_indices = np.where(seg_arr == 1)

        # Compute the size of the foreground along each dimension
        sizes = []
        for dim_indices in foreground_indices:
            dim_min = np.min(dim_indices)  # Minimum index in this dimension
            dim_max = np.max(dim_indices)  # Maximum index in this dimension
            size = dim_max - dim_min + 1   # Include both endpoints
            sizes.append(size)

        # Convert to NumPy array for convenience
        sizes = np.array(sizes)
        max_size = int(np.max(sizes) * (1+margin))

        subject_new = tio.CropOrPad(target_shape=max_size, mask_name="seg")(subject)
        
        return subject_new.img.tensor
    