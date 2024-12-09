# Copyright (c) Cyril Zakka.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from:
# Video-Dataset-Loading-Pytorch: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
# --------------------------------------------------------

import os
import os.path
import numpy as np
from PIL import Image
import torch
from typing import List, Union
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd

# Video Dataloader
class VideoRecord(object):
    """
    Video sample metadata.
    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: List of [path, start_index, end_index, label]
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame + 1
    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        return int(self._data[3])


class VideoFrameDataset(Dataset):
    """
    Sample a video from the dataset: the video is divided into NUM_SEGMENTS
    segments, while FRAMES_PER_SEGMENT consecutive frames are taken from each segment.
    Args:
        root_path: The root path in which video folders lie.
        annotationfile_path: The .csv annotation file containing the video paths and labels.
        num_segments: The number of segments the video should be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders.
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.
    """
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str='img_{:06d}.jpg',
                 transform = None,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')

    def _parse_annotationfile(self):
        df = pd.read_csv(self.annotationfile_path)
        self.video_list = [VideoRecord(x, self.root_path) for x in df.values.tolist()]

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                      f"but the dataloader is set up to load "
                      f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                      f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                      f"error when trying to load this video.\n")

    def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
        """
        For each segment, choose a start index from where frames
        are to be loaded from.
        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(self.num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments

            start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                      np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices

    def __getitem__(self, idx):
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.
        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]
        frame_start_indices = self._get_start_indices(record)
        return self._get(record, frame_start_indices)

    def _get(self, record, frame_start_indices):
        """
        Loads the frames of a video at the corresponding
        indices.
        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """

        frame_start_indices = frame_start_indices + record.start_frame
        images = list()

        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path, frame_index)
                if self.transform is not None:
                    image = self.transform(image)
                images.append(image)
                if frame_index < record.end_frame:
                    frame_index += 1
        return torch.stack(images).permute(1, 0, 2, 3), record.label

    def __len__(self):
        return len(self.video_list)

import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from monai import transforms


class NiftiDataset(Dataset):
    """
    Dataset for loading NIfTI files from the biobankT1 directory structure.
    """
    def __init__(self, root_path: str, transform=None):
        super(NiftiDataset, self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.subject_folders = self._get_subject_folders()

    def _get_subject_folders(self):
        folders = []
        for patient_id in os.listdir(self.root_path):
            patient_path = os.path.join(self.root_path, patient_id, 'T1', 'T1_brain.nii.gz')
            if os.path.exists(patient_path):
                folders.append(patient_path)
            else:
                print(f"Warning: Missing T1_brain.nii.gz for patient {patient_id}")
        return folders

    def __len__(self):
        return len(self.subject_folders)

    def __getitem__(self, idx):
        nifti_path = self.subject_folders[idx]
        try:
            nifti_image = nib.load(nifti_path)
        except Exception as e:
            print(f"Error loading NIfTI file at {nifti_path}: {e}")
            raise

        image_data = nifti_image.get_fdata()
        affine = nifti_image.affine

        # Convert to tensor and add channel dimension
        image_data = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)

        # Apply transforms, if any
        if self.transform:
            image_data = self.transform(image_data)

        label = 0  # Placeholder label
        return image_data, affine, label


def monai_transform(spatial_size=(160, 224, 160), scale_ratio=1., minv=0, maxv=1):
    return transforms.Compose([
        transforms.ResizeWithPadOrCrop(spatial_size=spatial_size, mode='minimum'),
        transforms.Orientation(axcodes='RAS'),
        transforms.Spacing(pixdim=scale_ratio),
        transforms.ScaleIntensity(minv=minv, maxv=maxv)
    ])


def monai_transform_test(spatial_size=(160, 224, 160), scale_ratio=1., minv=0, maxv=1):
    return transforms.Compose([
        transforms.ResizeWithPadOrCrop(spatial_size=spatial_size, mode='minimum'),
        transforms.Spacing(pixdim=scale_ratio),
        transforms.ScaleIntensity(minv=minv, maxv=maxv)
    ])

def build_dataloader(root_path, batch_size, num_workers, pin_mem, transform, is_train=True, distributed=False):
    """
    Build DataLoader for training or validation.
    """
    dataset = NiftiDataset(root_path=root_path, transform=transform)
    sampler = (
        torch.utils.data.RandomSampler(dataset) if not distributed
        else torch.utils.data.DistributedSampler(dataset)
    )
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=is_train
    )
    return dataloader


def save_as_nii(data, affine, output_path):
    """Save a 3D tensor as a NIfTI file."""
    if data.ndim == 4:
        data = data.squeeze(0)
    data_np = data.cpu().numpy()
    nii_image = nib.Nifti1Image(data_np, affine)
    nib.save(nii_image, output_path)
    print(f"Saved NIfTI file to {output_path}")


def save_random_slice_as_image(nii_path, output_dir, num_slices=5):
    """Save random slices from a NIfTI file as image files."""
    os.makedirs(output_dir, exist_ok=True)
    nii_image = nib.load(nii_path)
    data = nii_image.get_fdata()

    depth = data.shape[0]
    slice_indices = np.random.choice(range(depth), num_slices, replace=False)

    for i, slice_idx in enumerate(slice_indices):
        slice_data = data[slice_idx, :, :]
        plt.imshow(slice_data, cmap='gray')
        plt.axis('off')
        output_file = os.path.join(output_dir, f'slice_{i}_depth_{slice_idx}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved slice to {output_file}")


if __name__ == '__main__':
    class Args:
        data_path = './data'  # Root path for data
        output_path = './preprocessed_nii/'
        slice_output_path = './random_slices/'
        batch_size = 4
        num_workers = 4
        pin_mem = True
        distributed = False

    args = Args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.slice_output_path, exist_ok=True)

    # Build transform
    transform_test = monai_transform()

    dataset_test = NiftiDataset(root_path=os.path.join(args.data_path, 'test'), transform=transform_test)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                                                   pin_memory=args.pin_mem, drop_last=True)

    # Save the first 3 batches of NIfTI and random slices
    max_batches = 3  # Maximum number of batches to process
    for i, (data, affine, _) in enumerate(data_loader_test):
        if i >= max_batches:
            break

        for j in range(data.size(0)):  # Iterate over batch
            nii_file = os.path.join(args.output_path, f'preprocessed_sample_batch{i}_sample{j}.nii.gz')

            # Save NIfTI
            save_as_nii(data[j], affine[j].numpy(), nii_file)

            # Save random slices
            save_random_slice_as_image(nii_file, args.slice_output_path, num_slices=5)
