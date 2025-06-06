# File: run_phase2.py (Corrected)

import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator
from typing import Dict, Any

# --- 1. Define the MUE callback function ---
def mue_phase2_callback(step: int, timestep: int, latents: torch.Tensor, current_guidance_scale: float, callback_kwargs: Dict[str, Any]):
    """
    MUE's callback function for Phase 2, calculating and logging DIS.
    """
    model_name = callback_kwargs.get("model_name", "Unknown Model")
    dis_calculator: DISCalculator = callback_kwargs.get("dis_calculator_instance")
    # [FIXED] The pipeline passes the VAE with the key 'vae'
    vae = callback_kwargs.get("vae")

    if dis_calculator and vae:
        # Calculate DIS score
        dis_score = dis_calculator.calculate_dis(
            latents=latents,
            vae=vae, # [FIXED] Pass the vae instance with the correct keyword
            perturbation_scale=0.01,
            num_perturbations=5
        )
        print(f"[{model_name}] Step {step:02d} (Timestep {timestep:04d}): DIS Score = {dis_score:.6f}")
    else:
        print(f"[{model_name}] Step {step:02d} (Timestep {timestep:04d}): DIS Calculator or VAE not found in kwargs.")
    
    return current_guidance_scale # Callbacks that adapt guidance should return it

# --- 2. Main execution block ---
if __name__ == "__main__":
    print("Starting Phase 2: Diffusion Instability Signal (DIS) Test...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize our custom MUE pipeline
    mue_pipe = MUEDiffusionPipeline(device=device)
    print(f"Pipeline initialization complete.")

    # Initialize DISCalculator
    dis_calculator = DISCalculator(device=device)

    # --- 3. Define generation parameters ---
    prompt = "photorealistic image of a lone astronaut exploring a vibrant alien jungle, bioluminescent plants, misty atmosphere"
    negative_prompt = "blurry, low quality, distorted, bad art, cartoon, anime, ugly, deformed, text, oversaturated"
    seed = 99

    # --- 4. Define callback kwargs for context ---
    callback_kwargs_base = {
        "model_name": "Base", 
        "dis_calculator_instance": dis_calculator,
    }
    # [FIXED] The refiner callback will now work
    callback_kwargs_refiner = {
        "model_name": "Refiner", 
        "dis_calculator_instance": dis_calculator,
    }

    # --- 5. Perform generation ---
    start_gen_time = time.time()
    result = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps_base=25,
        num_inference_steps_refiner=5,
        initial_guidance_scale=7.5,
        seed=seed,
        callback_on_step_end=mue_phase2_callback,
        callback_on_step_end_kwargs_base=callback_kwargs_base,
        callback_on_step_end_kwargs_refiner=callback_kwargs_refiner
    )
    end_gen_time = time.time()
    
    # --- 6. Save and Report ---
    output_filename = "sdxl_mue_phase2_image.png"
    result['images'][0].save(output_filename)
    
    print(f"\nImage saved to {output_filename}")
    print(f"Total Generation Time: {end_gen_time - start_gen_time:.2f} seconds")
    print("\nPhase 2 Test Complete. You should now see DIS Scores from both Base and Refiner stages.")