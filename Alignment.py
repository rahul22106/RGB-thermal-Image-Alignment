import os
import cv2
import numpy as np
import torch
from glob import glob
from torchvision.transforms.functional import to_tensor
from kornia.feature import LoFTR

# Configuration
INPUT_DIR = "input_folder"                   
OUTPUT_DIR = "output_folder"                 
TARGET_SIZE = (640, 512)
INVERT_THERMAL = True
MIN_MATCHED_KEYPOINTS = 10

# Setup - create folders if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loftr = LoFTR(pretrained='outdoor').to(device).eval()


def get_base_name(thermal_path):
    return os.path.basename(thermal_path).replace("_T.JPG", "")


def preprocess_for_loftr(image_bgr, invert=False):
    """Convert BGR image to grayscale tensor, resized for LoFTR."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    resized = cv2.resize(gray, TARGET_SIZE)
    tensor = to_tensor(resized.astype(np.float32) / 255.).unsqueeze(0).to(device)
    return tensor, resized


def align_thermal_to_rgb():
    thermal_paths = glob(os.path.join(INPUT_DIR, "*_T.JPG"))

    if not thermal_paths:
        print(f"No thermal images found in {INPUT_DIR}")
        return

    for thermal_path in thermal_paths:
        base_name = get_base_name(thermal_path)
        rgb_path = os.path.join(INPUT_DIR, base_name + "_Z.JPG")

        if not os.path.exists(rgb_path):
            print(f"Missing RGB image for {base_name}")
            continue

        thermal_color = cv2.imread(thermal_path)
        rgb_color = cv2.imread(rgb_path)

        if thermal_color is None or rgb_color is None:
            print(f"Failed to read image(s) for {base_name}")
            continue

        # 1. Copy RGB image to output folder (unchanged)
        rgb_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_Z.JPG")
        cv2.imwrite(rgb_output_path, rgb_color)
        print(f"Saved RGB image: {rgb_output_path}")

        # 2. Resize both images to target size for alignment
        thermal_resized = cv2.resize(thermal_color, TARGET_SIZE)
        rgb_resized = cv2.resize(rgb_color, TARGET_SIZE)

        # 3. Create LoFTR inputs from resized images
        thermal_tensor, thermal_gray = preprocess_for_loftr(thermal_resized, invert=INVERT_THERMAL)
        rgb_tensor, rgb_gray = preprocess_for_loftr(rgb_resized, invert=False)

        with torch.no_grad():
            output = loftr({'image0': thermal_tensor, 'image1': rgb_tensor})

        kpts0 = output['keypoints0'].cpu().numpy()
        kpts1 = output['keypoints1'].cpu().numpy()

        if len(kpts0) < MIN_MATCHED_KEYPOINTS:
            print(f"Not enough matches for {base_name} ({len(kpts0)} points)")
            continue

        H, status = cv2.findHomography(kpts0, kpts1, cv2.RANSAC)
        if H is None:
            print(f"Homography estimation failed for {base_name}")
            continue

        # 4. Apply homography to the resized thermal image
        aligned_thermal = cv2.warpPerspective(thermal_resized, H, TARGET_SIZE)

        # 5. Save aligned thermal image
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_AT.JPG")
        cv2.imwrite(out_path, aligned_thermal)
        print(f"Saved aligned thermal image: {out_path} (size: {aligned_thermal.shape})")


if __name__ == "__main__":
    print("=" * 60)
    print("RGB-Thermal Image Alignment")
    print("=" * 60)
    print(f"Input folder: {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("-" * 60)
    align_thermal_to_rgb()
    print("-" * 60)
    print("Processing complete!")
    print(f"Check '{OUTPUT_DIR}' folder for results")