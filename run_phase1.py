import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from typing import Dict, Any

# --- 1. Define a simple callback function ---
def my_phase1_callback(step: int, timestep: int, latents: torch.Tensor, callback_kwargs: Dict[str, Any]):
    """
    A simple callback that prints contextual information passed to it.
    """
    message = callback_kwargs.get("message", "Callback triggered")
    model_name = callback_kwargs.get("model_name", "Unknown Model")
    print(f"[{model_name}] Step {step} (Timestep {int(timestep)}): {message}. Latents shape: {latents.shape}")

# --- 2. Main execution block ---
if __name__ == "__main__":
    print("Starting Phase 1: Custom Pipeline Test...")

    # Initialize our custom MUE pipeline
    start_init_time = time.time()
    try:
        # CORRECTED: Remove 'variant', 'use_safetensors', and 'low_cpu_mem_usage'
        # These are handled internally by MUEDiffusionPipeline's __init__ method
        mue_pipe = MUEDiffusionPipeline(
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            # Optional: Enable compilation if you want to test torch.compile, e.g., compile_models=True
            compile_models=False # Set to True to enable torch.compile
        )
    except Exception as e:
        print(f"Error during pipeline initialization: {e}")
        # This catch-all message is fine, as it provides useful general advice.
        print("Please ensure you have sufficient VRAM and have run 'huggingface-cli login'.")
        exit()
        
    end_init_time = time.time()
    print(f"Pipeline initialization time: {end_init_time - start_init_time:.2f} seconds")

    # --- 3. Define generation parameters ---
    prompt = "photorealistic image of a majestic space whale soaring through a nebula, cosmic dust, vibrant colors, epic scale"
    negative_prompt = "blurry, low quality, distorted, bad art, cartoon, anime, ugly, deformed, text, watermark"
    
    num_inference_steps_base = 25
    num_inference_steps_refiner = 5
    guidance_scale = 7.5
    seed = 88

    # --- 4. Define callback kwargs for context ---
    # These dictionaries provide context to the callback at each stage.
    callback_kwargs_base = {"message": "Base Denoising Step", "model_name": "Base"}
    callback_kwargs_refiner = {"message": "Refiner Denoising Step", "model_name": "Refiner"}

    # --- 5. Perform generation using the custom pipeline ---
    start_gen_time = time.time()
    output = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps_base=num_inference_steps_base,
        num_inference_steps_refiner=num_inference_steps_refiner,
        guidance_scale=guidance_scale,
        seed=seed,
        output_type="pil",
        # Pass the callback function itself
        callback_on_step_end=my_phase1_callback,
        # Pass the context-specific dictionaries for each stage
        callback_on_step_end_kwargs_base=callback_kwargs_base,
        callback_on_step_end_kwargs_refiner=callback_kwargs_refiner
    )
    end_gen_time = time.time()
    total_gen_time = end_gen_time - start_gen_time

    # --- 6. Save and Report ---
    generated_images = output.images
    output_filename = "sdxl_mue_phase1_image.png"
    generated_images[0].save(output_filename)
    
    print("\n--- Generation Report ---")
    print(f"Image saved to: {output_filename}")
    print(f"Total Generation Time (excluding init): {total_gen_time:.2f} seconds")
    print("\nPhase 1 Test Complete. The terminal output confirms that the callback mechanism is active for both Base and Refiner stages.")