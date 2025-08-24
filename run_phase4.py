# File: run_phase4_memory_optimized.py

import torch
import time
import gc
from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator
from gloss_module import GlossCalculator
from typing import Dict, Any

# --- MEMORY-OPTIMIZED MUE Parameters ---
MIN_GUIDANCE_SCALE = 4.0
MAX_GUIDANCE_SCALE = 12.0
DIS_STABLE_THRESHOLD = 0.002
DIS_UNSTABLE_THRESHOLD = 0.008
LAMBDA_INCREASE_RATE = 0.5
LAMBDA_DECREASE_RATE = 1.0

GLOSS_TARGET_CONCEPT = "sharp details, vibrant colors, clear focus, high contrast, photorealistic"
GLOSS_STRENGTH = 0.05  # Reduced from 0.08 for memory efficiency
GLOSS_GRADIENT_CLIP_NORM = 0.5
GLOSS_ACTIVE_START_STEP = 8  # Start later to save memory in early steps

# --- Memory Management Helper ---
def clear_gpu_cache():
    """Aggressively clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# --- 1. Memory-Optimized MUE Callback ---
def memory_optimized_mue_callback(step: int, timestep: int, latents: torch.Tensor, 
                                current_guidance_scale: float, callback_kwargs: Dict[str, Any]) -> float:
    """
    Memory-optimized callback that only applies DIS every few steps to reduce memory pressure
    """
    dis_calculator = callback_kwargs.get("dis_calculator_instance")
    vae = callback_kwargs.get("vae")
    
    # Only calculate DIS every 3 steps to reduce memory usage
    if step % 3 == 0 and dis_calculator and vae:
        try:
            with torch.cuda.amp.autocast():  # Use mixed precision
                dis_score = dis_calculator.calculate_dis(
                    latents=latents, 
                    vae=vae,
                    perturbation_scale=0.005,  # Reduced perturbation scale
                    num_perturbations=3        # Reduced from 5 to 3
                )
            
            print(f"[Step {step:02d}] DIS={dis_score:.5f} | λ_in={current_guidance_scale:.2f}", end="")
            
            if dis_score < DIS_STABLE_THRESHOLD:
                new_guidance_scale = current_guidance_scale + LAMBDA_INCREASE_RATE
                print(f" → Stable, λ_out={new_guidance_scale:.2f}")
            elif dis_score > DIS_UNSTABLE_THRESHOLD:
                new_guidance_scale = current_guidance_scale - LAMBDA_DECREASE_RATE
                print(f" → Unstable, λ_out={new_guidance_scale:.2f}")
            else:
                print(" → Normal")
                return current_guidance_scale
            
            return max(MIN_GUIDANCE_SCALE, min(new_guidance_scale, MAX_GUIDANCE_SCALE))
            
        except torch.cuda.OutOfMemoryError:
            print(f"[Step {step:02d}] DIS calculation OOM, skipping this step")
            clear_gpu_cache()
            return current_guidance_scale
    else:
        # Silent steps to reduce terminal spam
        if step % 5 == 0:
            print(f"[Step {step:02d}] λ={current_guidance_scale:.2f}")
    
    return current_guidance_scale

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    print("Starting Phase 4: MEMORY-OPTIMIZED Full MUE Test (Adaptive Lambda + Gloss)...")
    
    # Force clear any existing GPU memory
    clear_gpu_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing pipeline with memory optimizations...")
    
    # Initialize with more conservative settings
    mue_pipe = MUEDiffusionPipeline(
        device=device, 
        torch_dtype=torch.float16,  # Ensure FP16
        compile_models=False        # Disable compilation to save memory
    )
    
    print("Initializing calculators...")
    dis_calculator = DISCalculator(device=device, compile_models=False)
    gloss_calculator = GlossCalculator(device=device, compile_models=False)
    
    print("Pipeline, DIS, and Gloss Calculators initialized.")

    # --- 3. Conservative Generation Parameters ---
    prompt = "photo of a majestic lion in the african savanna, golden hour lighting"
    negative_prompt = "blurry, drawing, painting, cartoon, duplicate heads, deformed, low quality"
    seed = 2025

    # --- 4. Memory-Optimized Generation ---
    print("\nStarting memory-optimized generation...")
    clear_gpu_cache()  # Clear before generation
    
    start_gen_time = time.time()
    
    try:
        with torch.cuda.amp.autocast():  # Use automatic mixed precision
            result = mue_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=1024,  # Keep resolution high
                width=1024,
                num_inference_steps_base=20,      # Reduced from 25
                num_inference_steps_refiner=3,    # Reduced from 5
                initial_guidance_scale=7.0,       # Slightly lower
                seed=seed,
                # DIS parameters (memory optimized)
                callback_on_step_end=memory_optimized_mue_callback,
                callback_on_step_end_kwargs_base={
                    "dis_calculator_instance": dis_calculator
                },
                callback_on_step_end_kwargs_refiner={
                    "model_name": "Refiner", 
                    "dis_calculator_instance": dis_calculator
                },
                # Gloss parameters (memory optimized)
                gloss_calculator=gloss_calculator,
                gloss_target=GLOSS_TARGET_CONCEPT,
                gloss_strength=GLOSS_STRENGTH,
                gloss_gradient_clip_norm=GLOSS_GRADIENT_CLIP_NORM,
                gloss_active_start_step=GLOSS_ACTIVE_START_STEP
            )
        
        end_gen_time = time.time()
        
        # --- 5. Save and Report ---
        output_filename = "sdxl_mue_phase4_memory_optimized.png"
        result['images'][0].save(output_filename)
        
        print(f"\n✅ SUCCESS! Image saved to {output_filename}")
        print(f"Total Generation Time: {end_gen_time - start_gen_time:.2f} seconds")
        print("Phase 4 Memory-Optimized Test Complete!")
        
        # Print memory stats
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU Memory Used: {memory_used:.2f} GB")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Still getting OOM error: {e}")
        print("\nTrying even more aggressive memory optimization...")
        
        # Ultra-aggressive fallback
        clear_gpu_cache()
        
        print("Attempting with minimal settings...")
        try:
            result = mue_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=768,   # Reduced resolution
                width=768,
                num_inference_steps_base=15,  # Further reduced
                num_inference_steps_refiner=2,
                initial_guidance_scale=6.0,
                seed=seed,
                # Only Gloss, no DIS to save memory
                gloss_calculator=gloss_calculator,
                gloss_target=GLOSS_TARGET_CONCEPT,
                gloss_strength=0.03,  # Very light Gloss
                gloss_active_start_step=12,  # Very late start
                # No DIS callbacks
            )
            
            output_filename = "sdxl_mue_phase4_minimal.png"
            result['images'][0].save(output_filename)
            print(f"✅ Minimal version succeeded! Saved to {output_filename}")
            
        except Exception as final_e:
            print(f"❌ Final fallback failed: {final_e}")
            print("Your GPU might need even more aggressive optimization or a smaller model.")
    
    finally:
        # Clean up
        clear_gpu_cache()
        print("\nMemory cleanup completed.")
