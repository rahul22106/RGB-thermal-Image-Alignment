# RGB-Thermal Image Alignment

A Python tool for aligning thermal and RGB image pairs using deep learning-based feature matching with LoFTR (Local Feature Matching with Transformers).

## Overview

This tool automatically aligns thermal images with their corresponding RGB images by detecting matching keypoints and computing homography transformations. It's particularly useful for applications in computer vision, autonomous systems, and multi-spectral imaging where precise alignment between thermal and visible light images is crucial.

## Features

- **Automated Image Alignment**: Uses LoFTR for robust feature matching between thermal and RGB images
- **GPU Acceleration**: Leverages CUDA when available for faster processing
- **Batch Processing**: Processes multiple image pairs automatically
- **Configurable Parameters**: Easy customization of image sizes, inversion settings, and matching thresholds
- **Quality Control**: Validates homography estimation and minimum keypoint requirements

## Requirements

- Python 3.7+
- CUDA-capable GPU (optional, but recommended for faster processing)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Input Image Naming Convention

The tool expects image pairs with specific naming patterns:
- Thermal images: `<basename>_T.JPG`
- RGB images: `<basename>_Z.JPG`

For example:
- `image001_T.JPG` (thermal)
- `image001_Z.JPG` (RGB)

### Running the Alignment

1. Place your thermal and RGB image pairs in the `input_folder` directory

2. Run the script:
```bash
python Alignment.py
```

3. Aligned images will be saved in the `output_folder` directory:
   - Original RGB images: `<basename>_Z.JPG`
   - Aligned thermal images: `<basename>_AT.JPG`

### Configuration

You can adjust the following parameters in `Alignment.py`:

```python
INPUT_DIR = "input_folder"              # Input directory path
OUTPUT_DIR = "output_folder"            # Output directory path
TARGET_SIZE = (640, 512)                # Output image dimensions (width, height)
INVERT_THERMAL = True                   # Invert thermal image before matching
MIN_MATCHED_KEYPOINTS = 10              # Minimum keypoints required for alignment
```

## How It Works

1. **Image Loading**: Reads thermal and RGB image pairs from the input directory
2. **Preprocessing**: Resizes images and converts to grayscale for feature matching
3. **Feature Matching**: Uses LoFTR to detect corresponding keypoints between images
4. **Homography Estimation**: Computes transformation matrix using RANSAC
5. **Image Warping**: Applies transformation to align thermal image with RGB
6. **Output**: Saves both original RGB and aligned thermal images

## Technical Details

- **LoFTR Model**: Pretrained 'outdoor' model for robust feature matching
- **Homography Estimation**: RANSAC algorithm for outlier rejection
- **Image Processing**: OpenCV for image I/O and transformations
- **Deep Learning Framework**: PyTorch with optional CUDA support

## Output

The script produces:
- Unmodified RGB images copied to output folder
- Aligned thermal images warped to match RGB perspective
- Console output showing processing status and statistics

Example output:
```
============================================================
RGB-Thermal Image Alignment
============================================================
Input folder: input_folder
Output folder: output_folder
------------------------------------------------------------
Saved RGB image: output_folder/image001_Z.JPG
Saved aligned thermal image: output_folder/image001_AT.JPG (size: (512, 640, 3))
------------------------------------------------------------
Processing complete!
Check 'output_folder' folder for results
```

## Troubleshooting

**Not enough matches found:**
- Ensure thermal and RGB images have sufficient overlap
- Try adjusting `MIN_MATCHED_KEYPOINTS` to a lower value
- Check if `INVERT_THERMAL` setting is appropriate for your thermal images

**Homography estimation failed:**
- Images may have too little overlap or similarity
- Verify that image pairs correspond to the same scene
- Consider using images with more distinctive features

**Missing RGB image error:**
- Ensure RGB images follow the naming convention `<basename>_Z.JPG`
- Check that thermal and RGB pairs have matching base names

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LoFTR](https://github.com/zju3dv/LoFTR) for the feature matching model
- [Kornia](https://github.com/kornia/kornia) for the PyTorch computer vision library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Rahul Kumar Mishra

## Citation

If you use this tool in your research, please cite the LoFTR paper:

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={CVPR},
  year={2021}
}
```