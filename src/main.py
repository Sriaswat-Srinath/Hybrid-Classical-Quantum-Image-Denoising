#!/usr/bin/env python3
# ------------------------------------------------
# 14-qubit QFT-Grover denoising  +  OpenCL edge filter
# POCL / Intel / AMD / NVIDIA OpenCL – local PC
# ------------------------------------------------
import numpy as np, cv2, time, json
import pyopencl as cl
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import Aer
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ---------- helpers ---------------------------------------------------------
def img2vec(img):
    vec = img.astype(np.float32).flatten()
    vec /= np.linalg.norm(vec)
    return vec
def vec2img(vec, shape):
    im = vec.reshape(shape)
    im = (255 * im / im.max()).clip(0,255).astype(np.uint8)
    return im

# ---------- 1. load & noise -------------------------------------------------
clean = cv2.resize(cv2.imread('/home/user/Pictures/Screenshots/ai-generated-picture-of-a-tiger-walking-in-the-forest-photo.jpg', 0), (128,128))
noisy = clean + np.random.normal(0,25,clean.shape)
noisy = np.uint8(np.clip(noisy,0,255))
vec   = img2vec(noisy)
shape = clean.shape

# ---------- 2. quantum denoise (CPU) ---------------------------------------
# ---------- 2. quantum denoise (CPU) ---------------------------------------
def quantum_denoise(vec, shape=(64, 64)):
    n = int(np.ceil(np.log2(np.prod(shape))))   # 14 for 128×128
    sv  = np.pad(vec, (0, (1 << n) - len(vec)), 'constant')
    qc  = QuantumCircuit(n)
    qc.set_statevector(sv)
    qc.append(QFTGate(n), range(n))

    # oracle |00..0>
    qc.x(range(n)); qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1); qc.x(range(n))
    # diffusion
    qc.h(range(n)); qc.x(range(n)); qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1); qc.x(range(n)); qc.h(range(n))
    # 2 iterations
    for _ in range(2):
        qc.x(range(n)); qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1); qc.x(range(n))
        qc.h(range(n)); qc.x(range(n)); qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1); qc.x(range(n)); qc.h(range(n))

    qc.append(QFTGate(n).inverse(), range(n))
    sim  = Aer.get_backend('statevector_simulator')
    sv   = sim.run(transpile(qc, sim)).result().get_statevector()
    prob = np.abs(sv.data)[:np.prod(shape)].reshape(shape)   
    return prob

# ---------- usage ------------------------------------------------------------
shape = (64, 64)              # ≤ 16384 pixels
prob_q = quantum_denoise(vec, shape)
print('✅ 12-qubit quantum denoised')

# ---------- 3. OpenCL edge filter -------------------------------------------
# ---- choose first GPU / CPU OpenCL platform
platform = cl.get_platforms()[0]
device   = platform.get_devices()[0]
ctx      = cl.Context([device])
queue    = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# ---- kernel source
src = '''
__kernel void sobel_and_canny(__global const uchar *in,
                              __global uchar *out_sobel,
                              __global uchar *out_canny,
                              const int w, const int h,
                              const float lowFrac,
                              const float highFrac) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < 1 || y < 1 || x >= w-1 || y >= h-1) return;

    // 1. Sobel gradients
    int gx = -in[(y-1)*w + x-1] + in[(y-1)*w + x+1]
             -2*in[ y   *w + x-1] +2*in[ y   *w + x+1]
             -in[(y+1)*w + x-1] + in[(y+1)*w + x+1];
    int gy = -in[(y-1)*w + x-1] -2*in[(y-1)*w + x] -in[(y-1)*w + x+1]
             +in[(y+1)*w + x-1] +2*in[(y+1)*w + x] +in[(y+1)*w + x+1];
    int mag = (int)sqrt((float)(gx*gx + gy*gy));
    out_sobel[y*w + x] = convert_uchar_sat(mag);

    // 2. adaptive Canny thresholds
    float median = 127.0f;          // quick surrogate
    float high = median * highFrac;
    float low  = median * lowFrac;
    out_canny[y*w + x] = (mag > high) ? 255 : ((mag > low) ? 128 : 0);
}
'''
prg   = cl.Program(ctx, src).build()

prob_u8 = (255 * prob_q / prob_q.max()).astype(np.uint8)
h, w    = prob_u8.shape
d_in    = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,  prob_u8.nbytes)
d_sobel = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, prob_u8.nbytes)   # 1st output
d_canny = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, prob_u8.nbytes)   # 2nd output
cl.enqueue_copy(queue, d_in, prob_u8)
# ---------- 4.  metrics  -----------------------------------------------------
# cast quantum output to uint8 & verify shape
quant_u8 = vec2img(prob_q, shape)          # 0-255 uint8
assert clean.shape == quant_u8.shape, f"Shape mismatch: {clean.shape} vs {quant_u8.shape}"

psnr_val = psnr(clean, quant_u8)
ssim_val = ssim(clean, quant_u8)

evt = prg.sobel_and_canny(queue, prob_u8.shape, None,
                          d_in, d_sobel, d_canny,
                          np.int32(w), np.int32(h),
                          np.float32(0.33), np.float32(0.66))
evt.wait()
opencl_ms = (evt.profile.end - evt.profile.start)*1e-6

sobel_edges = np.empty_like(prob_u8); cl.enqueue_copy(queue, sobel_edges, d_sobel)
canny_edges = np.empty_like(prob_u8); cl.enqueue_copy(queue, canny_edges, d_canny)
print('✅ OpenCL Sobel + Canny  |  GPU time:', round(opencl_ms,3), 'ms')

# ---------- 4. metrics -------------------------------------------------------
psnr_val = psnr(clean, vec2img(prob_q, shape))   # vec2img already returns uint8
ssim_val = ssim(clean, vec2img(prob_q, shape))

out = {
    "PSNR (dB)": round(psnr_val,2),
    "SSIM": round(ssim_val,3),
    "OpenCL time (ms)": round(opencl_ms,2)
}
print(json.dumps(out, indent=2))

# ---------- 5. quick preview -----------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3,figsize=(12,4))
ax[0].imshow(noisy, cmap='gray'); ax[0].set_title('Noisy')
ax[1].imshow(vec2img(prob_q, shape), cmap='gray'); ax[1].set_title('Denoised')
ax[2].imshow(prob_u8, cmap='gray'); ax[2].set_title('OpenCL edges')
for a in ax: a.axis('off')
plt.tight_layout()
plt.savefig("hybridquan_output.png")

# ---------- 5.  four-panel PNG export  -------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].imshow(noisy, cmap='gray')
ax[0].set_title('Noisy'); ax[0].axis('off')

ax[1].imshow(vec2img(prob_q, shape), cmap='gray')
ax[1].set_title('Quantum denoised'); ax[1].axis('off')

ax[2].imshow(sobel_edges, cmap='gray')
ax[2].set_title('OpenCL Sobel'); ax[2].axis('off')

ax[3].imshow(canny_edges, cmap='gray')
ax[3].set_title('OpenCL Canny'); ax[3].axis('off')

plt.tight_layout()
out_file = '/home/user/hybrid-quantum-denoise-opencl/opencl_impl'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print('Saved →', out_file)
plt.show()