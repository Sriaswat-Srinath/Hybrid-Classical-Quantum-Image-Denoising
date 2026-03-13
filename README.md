# Hybrid Quantum-Classical Image Denoising and Edge Detection

This repository implements a hybrid quantum-classical pipeline for image processing. It leverages a 14-qubit Quantum Fourier Transform (QFT) combined with Grover's search algorithm for image denoising, followed by an OpenCL-accelerated classical pipeline for Sobel and adaptive Canny edge detection.

## 🚀 Overview

The pipeline operates in two distinct phases:

1. **Quantum Denoising (CPU/Simulator)**:
   - Uses a 14-qubit quantum circuit to encode 128×128 image patches (16,384 amplitudes).
   - The circuit applies the Quantum Fourier Transform (QFT).
   - High-frequency noise components are marked using an oracle.
   - Grover diffusion operator (2 iterations) suppresses the high-frequency components.
   - Inverse QFT brings the state back to the computational basis.
   - **Result**: A low-pass filtered, denoised image (output as probability map converted to `uint8`).

2. **OpenCL Edge Detection (GPU/CPU)**:
   - The quantum-denoised probability distribution is fed into a highly optimized OpenCL kernel.
   - Performs Sobel magnitude gradient calculation and adaptive Canny edge detection in **a single kernel launch**.
   - Cross-platform: Automatically runs on the primary OpenCL platform (POCL, Intel, AMD, or NVIDIA).

## 📁 Repository Structure

```text
hybrid-quantum-denoise-opencl/
├── assets/                  # Example outputs and diagrams
├── examples/                # Example input images (e.g., tiger.jpg)
├── notebooks/               # Jupyter Notebooks exploring the CUDA/Quantum implementation
├── outputs/                 # Directory where generated images are saved
├── src/                     # Source code directory
│   └── main.py              # Main execution script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Prerequisites

* Python 3.8+
* OpenCL runtime for your CPU or GPU (e.g., `pocl-opencl-icd`, `ocl-icd-libopencl1` on Ubuntu/Debian)

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### Ubuntu / Debian Dependency Installation
If you do not have OpenCL platforms installed and want to run via a CPU runtime, you can install POCL:
```bash
sudo apt update
sudo apt install pocl-opencl-icd ocl-icd-libopencl1
```

## 💻 Quick Start

Run the main pipeline using the provided example image:

```bash
python src/main.py examples/tiger.jpg
```

**What happens?**
1. The script loads `examples/tiger.jpg`, resizes it to 128x128, and adds artificial Gaussian noise.
2. The quantum simulator runs the 14-qubit QFT-Grover circuit to denoise the image.
3. The OpenCL kernel processes the denoised image to extract Sobel and Canny edges.
4. An output image containing a 4-panel comparison (Noisy, Quantum denoised, OpenCL Sobel, OpenCL Canny) is saved to `outputs/hybridquan_sobel_canny.png`.
5. Terminal metrics (PSNR, SSIM, and OpenCL execution time) are printed.

## 📊 Metrics

The script outputs quantitative image quality metrics:
* **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates noise reduction.
* **SSIM (Structural Similarity Index)**: Measures perceptual similarity after denoising.
* **OpenCL time (ms)**: Profiles the GPU edge-detection kernel's performance.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page and submit pull requests.
