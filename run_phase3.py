# File: run_phase3.py (Corrected)

import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator # [FIXED] Import DISCalculator
from typing import Dict, Any

# --- MUE Parameters ---
MIN_GUIDANCE_SCALE = 4.0
MAX_GUIDANCE_SCALE = 12.0
DIS_STABLE_THRESHOLD = 0.002
DIS_UNSTABLE_THRESHOLD = 0.008
LAMBDA_INCREASE_RATE = 0.5
LAMBDA_DECREASE_RATE = 1.0

# --- 1. Define the MUE callback function ---
def mue_phase3_callback(step: int, timestep: int, latents: torch.Tensor, current_guidance_scale: float, callback_kwargs: Dict[str, Any]) -> float:
    model_name = callback_kwargs.get("model_name", "Unknown Model")
    dis_calculator = callback_kwargs.get("dis_calculator_instance") # [FIXED] Key name to be consistent
    vae = callback_kwargs.get("vae")
    new_guidance_scale = current_guidance_scale

    if dis_calculator and vae:
        dis_score = dis_calculator.calculate_dis(latents=latents, vae=vae)
        print(f"[{model_name}] Step {step:02d} (t={timestep:04d}): DIS={dis_score:.6f} | In λ={current_guidance_scale:.2f}", end="")

        if dis_score < DIS_STABLE_THRESHOLD:
            new_guidance_scale += LAMBDA_INCREASE_RATE
            print(f" -> Stable, Out λ={new_guidance_scale:.2f}")
        elif dis_score > DIS_UNSTABLE_THRESHOLD:
            new_guidance_scale -= LAMBDA_DECREASE_RATE
            print(f" -> Unstable, Out λ={new_guidance_scale:.2f}")
        else:
            print(" -> Normal")
        
        return max(MIN_GUIDANCE_SCALE, min(new_guidance_scale, MAX_GUIDANCE_SCALE))
    
    print(f"[{model_name}] Step {step:02d}: DIS Calculator or VAE not found. Lambda remains {current_guidance_scale:.2f}")
    return current_guidance_scale

# --- 2. Main execution block ---
if __name__ == "__main__":
    print("Starting Phase 3: Adaptive Lambda Modulation Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mue_pipe = MUEDiffusionPipeline(device=device)
    # [FIXED] Instantiate the required DISCalculator
    dis_calculator = DISCalculator(device=device)
    print("Pipeline and DIS Calculator initialized.")

    # --- 3. Define generation parameters ---
    prompt = "a highly detailed hyperrealistic portrait of an elderly wizard reading an ancient glowing spellbook in a dimly lit, cozy library, dust motes in the air, volumetric lighting"
    negative_prompt = "blurry, low quality, distorted, bad art, cartoon, anime, ugly, deformed, text, oversaturated, youth"
    seed = 100

    # --- 4. Define callback kwargs ---
    # [FIXED] Pass the dis_calculator instance to the callback
    callback_kwargs = {"model_name": "Base/Refiner", "dis_calculator_instance": dis_calculator}

    # --- 5. Perform generation ---
    start_gen_time = time.time()
    result = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        initial_guidance_scale=8.0,
        seed=seed,
        callback_on_step_end=mue_phase3_callback,
        callback_on_step_end_kwargs_base={"model_name": "Base", "dis_calculator_instance": dis_calculator},
        callback_on_step_end_kwargs_refiner={"model_name": "Refiner", "dis_calculator_instance": dis_calculator},
    )
    end_gen_time = time.time()

    # --- 6. Save and Report ---
    output_filename = "sdxl_mue_phase3_image.png"
    result['images'][0].save(output_filename)
    
    print(f"\nImage saved to {output_filename}")
    print(f"Total Generation Time: {end_gen_time - start_gen_time:.2f} seconds")
    print("\nPhase 3 Test Complete. Check terminal for adaptive lambda changes from both stages!")