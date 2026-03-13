# Hybrid 14-Qubit QFT-Grover + OpenCL Sobel/Canny  
**Local GPU / CPU – 128×128 patches 

---

## What it does
1. **CPU**: 14-qubit quantum circuit  
   - Amplitude-encode 128×128 patch (16 384 amplitudes)  
   - QFT → mark high-freq oracle → 2 Grover iterations → inverse QFT  
   - **Result**: low-pass denoised image (uint8)
2. **OpenCL**: edge detection on the quantum output  
   - Sobel magnitude + adaptive Canny in **one kernel launch**  
   - Works with **POCL**, Intel, AMD, NVIDIA ICDs

---

## Quick start (local PC)
```bash
# 1. clone
git clone &lt;your-repo&gt;
cd hybrid-quantum-denoise-opencl

# 2. deps
sudo apt install pocl-opencl-icd ocl-icd-libopencl1   # Ubuntu / Debian
pip install -r requirements.txt

# 3. run
python src/main.py examples/tiger.jpg
# → outputs/hybrid_128_sobel_canny.png
