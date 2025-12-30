# Hybrid Underwater Image Enhancement Using CNN Refinement

## Overview
This project presents a software-based framework for underwater image enhancement using a hybrid approach that combines classical image processing techniques with a convolutional neural network (CNN). The goal is to improve visual quality of underwater images affected by color distortion, low contrast, and haze.

## Motivation
Underwater images suffer from degradation due to light absorption and scattering in water. Traditional enhancement techniques alone are often insufficient, while deep learning models require large datasets. This project explores a hybrid solution that balances effectiveness and computational simplicity.

## Methodology
The system follows a multi-stage enhancement pipeline:
1. Input underwater image acquisition
2. Classical preprocessing (color correction and contrast enhancement)
3. CNN-based refinement for noise reduction and texture preservation
4. Generation of enhanced output image

## Tools and Technologies
- Python 3.9
- OpenCV
- PyTorch
- NumPy

## Dataset
- Public underwater image datasets
- Custom underwater and personal images for testing  
(*Datasets are not included in the repository due to size constraints.*)

## Results
The model produces visually improved images with enhanced contrast and reduced haze. Experimental results demonstrate the strengths and limitations of shallow CNN architectures when applied to underwater image enhancement.

## Limitations
- Shallow CNN architecture
- Limited training data
- Color distortion may occur for non-underwater images

## Future Improvements
- Integration of hybrid preprocessing (CLAHE, White Balance)
- Training on paired datasets such as UIEB
- Use of deeper CNN architectures
- Quantitative evaluation using PSNR and SSIM

## How to Run
1. Clone the repository
2. Create and activate a Python virtual environment
3. Install required dependencies
4. Train the model using `train.py`
5. Test the model using `test_model_on_image.py`

## Author
Rahul Khedekar

