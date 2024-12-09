import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from models_mri import mae3d_vit_base_patch16
from utils.dataloader import NiftiDataset, monai_transform_test

def load_model(img_size, model_path, device):
    """
    Load the pretrained model from a checkpoint.
    """
    model = mae3d_vit_base_patch16(img_size = img_size, norm_pix_loss=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"Model loaded from {model_path}")
    return model

def visualize_3d_slices(input_data, reconstructed_data, output_dir, prefix="sample"):
    """
    Visualize and save several slices from the input and reconstructed volumes.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_slices = input_data.shape[-1]
    slice_indices = np.linspace(0, num_slices - 1, 5, dtype=int)  # Visualize 5 evenly spaced slices

    for idx, slice_idx in enumerate(slice_indices):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(input_data[:, :, slice_idx], cmap='gray')
        axs[0].set_title(f"Input Slice {slice_idx}")
        axs[1].imshow(reconstructed_data[:, :, slice_idx], cmap='gray')
        axs[1].set_title(f"Reconstructed Slice {slice_idx}")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{prefix}_slice_{idx}.png")
        plt.savefig(save_path)
        print(f"Saved slice visualization to {save_path}")
        plt.close()

def process_nifti_files(model, data_loader, output_dir, device):
    """
    Process all samples in the DataLoader using the model and save results.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (input_tensor, affine, _) in enumerate(data_loader):
            input_tensor = input_tensor.to(device)

            # Perform inference
            total_loss, pred, _ = model(input_tensor, mask_ratio=0.7)
            reconstructed_data = model.unpatchify3D(pred).squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel
            print("Reconstructed data shape:", reconstructed_data.shape)

            # Load input data for visualization
            input_data = input_tensor.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel
            print("Input data shape:", input_data.shape)

            # Visualize and save slices
            visualize_3d_slices(input_data, reconstructed_data, output_dir, prefix=f"sample_{idx}")

            # Save reconstructed volume as NIfTI
            output_nii_path = os.path.join(output_dir, f"reconstructed_sample_{idx}.nii.gz")
            reference_affine = nib.load(data_loader.dataset.subject_folders[idx]).affine
            save_reconstructed_nii(reconstructed_data, reference_affine, output_nii_path)

def save_reconstructed_nii(output_data, reference_affine, output_path):
    """
    Save a 3D numpy array as a .nii.gz file with the reference affine.
    """
    output_nii = nib.Nifti1Image(output_data, reference_affine)
    nib.save(output_nii, output_path)
    print(f"Saved reconstructed volume to {output_path}")

def main():
    # Configurations
    model_path = "checkpoint-49.pth"  # 替换为你的模型路径
    data_path = "./data/test"  # 替换为验证数据路径
    output_dir = "./output_visualizations"
    img_size = (160, 224, 160)
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(img_size, model_path, device)

    # Prepare data
    transform = monai_transform_test(spatial_size = img_size)
    dataset = NiftiDataset(root_path=data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Process and visualize data
    process_nifti_files(model, data_loader, output_dir, device)

if __name__ == "__main__":
    main()
