# File: run_phase1.py (Corrected)

import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from typing import Dict, Any

# --- 1. Define a simple callback function ---
def my_phase1_callback(step: int, timestep: int, latents: torch.Tensor, current_guidance_scale: float, callback_kwargs: Dict[str, Any]):
    """
    A simple callback that prints contextual information passed to it.
    It must accept `current_guidance_scale` as an argument from the pipeline.
    """
    model_name = callback_kwargs.get("model_name", "Unknown Model")
    print(f"[{model_name}] Step {step:02d} (Timestep {int(timestep):04d}): Latents shape: {latents.shape}, Current λ: {current_guidance_scale:.2f}")
    # Callbacks should return the guidance scale, even if it's unchanged.
    return current_guidance_scale

# --- 2. Main execution block ---
if __name__ == "__main__":
    print("Starting Phase 1: Custom Pipeline Test...")

    start_init_time = time.time()
    try:
        # [SIMPLIFIED] The pipeline now handles internal setup. We only provide high-level choices.
        mue_pipe = MUEDiffusionPipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16,
            compile_models=False 
        )
    except Exception as e:
        print(f"Error during pipeline initialization: {e}")
        print("Please ensure you have sufficient VRAM and have run 'huggingface-cli login'.")
        exit()
        
    end_init_time = time.time()
    print(f"Pipeline initialization time: {end_init_time - start_init_time:.2f} seconds")

    # --- 3. Define generation parameters ---
    prompt = "photorealistic image of a majestic space whale soaring through a nebula, cosmic dust, vibrant colors, epic scale"
    negative_prompt = "blurry, low quality, distorted, bad art, cartoon, anime, ugly, deformed, text, watermark"
    
    # --- 4. Define callback kwargs for context ---
    callback_kwargs_base = {"model_name": "Base"}
    callback_kwargs_refiner = {"model_name": "Refiner"}

    # --- 5. Perform generation using the custom pipeline ---
    start_gen_time = time.time()
    output = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps_base=25,
        num_inference_steps_refiner=5,
        # [FIXED] The parameter name is 'initial_guidance_scale'
        initial_guidance_scale=7.5,
        seed=88,
        output_type="pil",
        callback_on_step_end=my_phase1_callback,
        callback_on_step_end_kwargs_base=callback_kwargs_base,
        callback_on_step_end_kwargs_refiner=callback_kwargs_refiner
    )
    end_gen_time = time.time()
    
    # --- 6. Save and Report ---
    generated_images = output["images"]
    output_filename = "sdxl_mue_phase1_image.png"
    generated_images[0].save(output_filename)
    
    print("\n--- Generation Report ---")
    print(f"Image saved to: {output_filename}")
    print(f"Total Generation Time (excluding init): {end_gen_time - start_gen_time:.2f} seconds")
    print("\nPhase 1 Test Complete. The terminal output should now correctly show callbacks for both stages.")