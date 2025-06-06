import torch
import time
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
#from diffusers.utils import export_to_image

print("Starting Phase 0: Baseline Inference Setup...")

# --- 1. Configuration ---
# Set device to CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable low CPU memory usage for model loading if on GPU.
# This prevents models from temporarily loading to CPU RAM before moving to GPU.
# This is a good practice for larger models even on 40GB A100 to avoid temporary memory spikes.
# If you encounter issues (e.g., 'half' being non-contiguous or error during loading),
# you might need to temporarily remove this for specific problematic models, but it's generally safe.
low_cpu_mem_usage = True if device == "cuda" else False

# Prompts for generation
prompt = "photorealistic image of a futuristic city at sunset, highly detailed, cyberpunk aesthetic, neon lights, flying cars"
negative_prompt = "low quality, bad anatomy, worst quality, blurry, low resolution, error, extra digits, cropped, ugly, distorted, noise, grain, cartoon, anime, drawing, illustration"

# Generation parameters
num_inference_steps_base = 25 # Typical for base model
num_inference_steps_refiner = 5 # Typical for refiner
guidance_scale = 7.5 # Standard CFG scale
seed = 42 # For reproducibility

# Set up generator for reproducibility
generator = torch.Generator(device=device).manual_seed(seed)

# --- 2. Load Optimized Models ---
print("Loading SDXL Base model...")
# Load SDXL Base pipeline
pipe_base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, # Use FP16 for efficiency
    variant="fp16",           # Load the FP16 variant of weights
    use_safetensors=True,
    low_cpu_mem_usage=low_cpu_mem_usage # Helps with large models
).to(device)

print("Loading SDXL Refiner model...")
# Load SDXL Refiner pipeline
pipe_refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    low_cpu_mem_usage=low_cpu_mem_usage
).to(device)

print("Loading optimized VAE...")
# Load and set optimized VAE (e.g., from madebyollin)
# This VAE has fixes for FP16 and often provides better quality.
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=low_cpu_mem_usage
).to(device)

# Attach the optimized VAE to both pipelines
pipe_base.vae = vae
pipe_refiner.vae = vae

# --- 3. Configure Scheduler with Stability Fixes ---
print("Configuring DPM++ Scheduler with stability fixes...")
# Using DPMSolverMultistepScheduler with recommended stability parameters
pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True, euler_at_final_sigma=True)
pipe_refiner.scheduler = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True, euler_at_final_sigma=True)

# --- 4. Enable Performance Optimizations ---
# PyTorch 2.0+ Native Scaled Dot Product Attention (SDPA)
# This is usually enabled by default if torch 2.x is used, but explicit context is useful.
# pipe_base.unet.config.enable_sdpa = True # Usually already set by default in config
# pipe_refiner.unet.config.enable_sdpa = True

# --- 5. Perform Two-Stage Inference ---
print(f"Starting generation with Base model ({num_inference_steps_base} steps)...")
start_time = time.time()

# Generate latent from base model for refiner input
image_base_latents = pipe_base(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps_base,
    denoising_end=0.8, # Denoise the last 20% with the refiner
    guidance_scale=guidance_scale,
    generator=generator,
    output_type="latent" # Output latents for refiner
).images

print(f"Base model finished. Latents shape: {image_base_latents.shape}")
print(f"Starting refinement with Refiner model ({num_inference_steps_refiner} steps)...")

# Refine the latents
image_final = pipe_refiner(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps_refiner,
    denoising_start=0.8, # Start denoising from 80% noise (where base left off)
    image=image_base_latents, # Pass latents from base model
    guidance_scale=guidance_scale,
    generator=generator,
    output_type="pil" # Output PIL image
).images[0] # Get the first (and only) image

end_time = time.time()
total_time = end_time - start_time

# --- 6. Save and Report ---
output_filename = "sdxl_baseline_image.png"
image_final.save(output_filename)
print(f"\nImage saved to {output_filename}")
print(f"Total Inference Time: {total_time:.2f} seconds")

# Optional: Print VRAM usage at the end (requires running nvidia-smi separately or via subprocess)
print("\nCheck VRAM usage with `nvidia-smi` in a separate terminal.")