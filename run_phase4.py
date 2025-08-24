# File: run_phase4.py (Corrected)
mue_pipe = MUEDiffusionPipeline(device=device)
mue_pipe.enable_model_cpu_offload() # Add this line
import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator   # [FIXED] Import DISCalculator
from gloss_module import GlossCalculator # [FIXED] Import GlossCalculator
from typing import Dict, Any

# --- MUE Parameters ---
MIN_GUIDANCE_SCALE = 4.0
MAX_GUIDANCE_SCALE = 12.0
DIS_STABLE_THRESHOLD = 0.002
DIS_UNSTABLE_THRESHOLD = 0.008
LAMBDA_INCREASE_RATE = 0.5
LAMBDA_DECREASE_RATE = 1.0

GLOSS_TARGET_CONCEPT = "sharp details, vibrant colors, clear focus, high contrast, photorealistic"
GLOSS_STRENGTH = 0.08
GLOSS_GRADIENT_CLIP_NORM = 0.5
GLOSS_ACTIVE_START_STEP = 5

# --- 1. MUE Callback for DIS/Adaptive Lambda ---
def mue_callback(step: int, timestep: int, latents: torch.Tensor, current_guidance_scale: float, callback_kwargs: Dict[str, Any]) -> float:
    # This callback is now only for DIS, Gloss is handled by the pipeline directly
    dis_calculator = callback_kwargs.get("dis_calculator_instance")
    vae = callback_kwargs.get("vae")
    
    if dis_calculator and vae:
        dis_score = dis_calculator.calculate_dis(latents=latents, vae=vae)
        print(f"[Base] Step {step:02d} | DIS={dis_score:.5f} | In λ={current_guidance_scale:.2f}", end="")
        
        if dis_score < DIS_STABLE_THRESHOLD:
            new_guidance_scale = current_guidance_scale + LAMBDA_INCREASE_RATE
            print(f" -> Stable, Out λ={new_guidance_scale:.2f}")
        elif dis_score > DIS_UNSTABLE_THRESHOLD:
            new_guidance_scale = current_guidance_scale - LAMBDA_DECREASE_RATE
            print(f" -> Unstable, Out λ={new_guidance_scale:.2f}")
        else:
            print(" -> Normal")
        
        return max(MIN_GUIDANCE_SCALE, min(new_guidance_scale, MAX_GUIDANCE_SCALE))
    
    return current_guidance_scale

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    print("Starting Phase 4: Full MUE Test (Adaptive Lambda + Gloss)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mue_pipe = MUEDiffusionPipeline(device=device)
    # [FIXED] Instantiate both DIS and Gloss calculators
    dis_calculator = DISCalculator(device=device)
    gloss_calculator = GlossCalculator(device=device)
    print("Pipeline, DIS, and Gloss Calculators initialized.")

    # --- 3. Generation Parameters ---
    prompt = "photo of a majestic lion in the african savanna, sunset lighting"
    negative_prompt = "blurry, drawing, painting, cartoon, duplicate heads, deformed"
    seed = 2025

    # --- 4. Define Callback Kwargs ---
    # [FIXED] Pass the dis_calculator instance
    callback_kwargs = {"dis_calculator_instance": dis_calculator}

    # --- 5. Perform Generation with Full MUE ---
    start_gen_time = time.time()
    result = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        initial_guidance_scale=8.0,
        seed=seed,
        # DIS parameters
        callback_on_step_end=mue_callback,
        callback_on_step_end_kwargs_base=callback_kwargs,
        # Note: Gloss is not applied in refiner by this design, but DIS is.
        callback_on_step_end_kwargs_refiner={"model_name": "Refiner", "dis_calculator_instance": dis_calculator},
        # Gloss parameters
        gloss_calculator=gloss_calculator, # [FIXED] Pass the gloss_calculator instance
        gloss_target=GLOSS_TARGET_CONCEPT,
        gloss_strength=GLOSS_STRENGTH,
        gloss_gradient_clip_norm=GLOSS_GRADIENT_CLIP_NORM,
        gloss_active_start_step=GLOSS_ACTIVE_START_STEP
    )
    end_gen_time = time.time()

    # --- 6. Save and Report ---
    output_filename = "sdxl_mue_phase4_image.png"
    result['images'][0].save(output_filename)
    
    print(f"\nImage saved to {output_filename}")
    print(f"Total Generation Time: {end_gen_time - start_gen_time:.2f} seconds")
    print("\nPhase 4 Test Complete. Both DIS/Lambda and Gloss were active.")
