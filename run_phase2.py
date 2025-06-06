import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator # Import our DISCalculator
from diffusers.utils import export_to_image
from typing import Dict, Any

# --- 1. Define the MUE callback function ---
def mue_phase2_callback(step: int, timestep: int, latents: torch.Tensor, callback_kwargs: Dict[str, Any]):
    """
    MUE's callback function for Phase 2, calculating and logging DIS.
    """
    model_name = callback_kwargs.get("model_name", "Unknown Model")
    dis_calculator: DISCalculator = callback_kwargs.get("dis_calculator_instance")
    vae_decoder = callback_kwargs.get("vae_decoder") # Get the VAE instance passed from pipeline

    if dis_calculator and vae_decoder:
        # Calculate DIS score
        # Using a slightly higher perturbation_scale for initial visibility of effect
        dis_score = dis_calculator.calculate_dis(
            latents=latents,
            vae_decoder=vae_decoder,
            perturbation_scale=0.01, # Increased for clearer signal in early stages
            num_perturbations=5
        )
        print(f"[{model_name}] Step {step} (Timestep {timestep}): DIS Score = {dis_score:.6f}")
    else:
        print(f"[{model_name}] Step {step} (Timestep {timestep}): DIS Calculator or VAE not found in kwargs.")

# --- 2. Main execution block ---
if __name__ == "__main__":
    print("Starting Phase 2: Diffusion Instability Signal (DIS) Test...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize our custom MUE pipeline
    start_init_time = time.time()
    mue_pipe = MUEDiffusionPipeline(
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device=device
    )
    end_init_time = time.time()
    print(f"Pipeline initialization time: {end_init_time - start_init_time:.2f} seconds")

    # Initialize DISCalculator
    dis_calculator = DISCalculator(device=device)

    # --- 3. Define generation parameters ---
    prompt = "photorealistic image of a lone astronaut exploring a vibrant alien jungle, bioluminescent plants, misty atmosphere"
    negative_prompt = "blurry, low quality, distorted, bad art, cartoon, anime, ugly, deformed, text, oversaturated"
    
    num_inference_steps_base = 25
    num_inference_steps_refiner = 5
    guidance_scale = 7.5
    seed = 99 # Another new seed

    # --- 4. Define callback kwargs for context ---
    # Pass the DISCalculator instance and other context
    callback_kwargs_base = {
        "model_name": "Base", 
        "dis_calculator_instance": dis_calculator,
        # vae_decoder will be added by MUEDiffusionPipeline automatically
    }
    callback_kwargs_refiner = {
        "model_name": "Refiner", 
        "dis_calculator_instance": dis_calculator,
        # vae_decoder will be added by MUEDiffusionPipeline automatically
    }

    # --- 5. Perform generation using the custom pipeline ---
    start_gen_time = time.time()
    generated_images = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps_base=num_inference_steps_base,
        num_inference_steps_refiner=num_inference_steps_refiner,
        guidance_scale=guidance_scale,
        seed=seed,
        output_type="pil",
        callback_on_step_end=mue_phase2_callback,
        callback_on_step_end_kwargs_base=callback_kwargs_base,
        callback_on_step_end_kwargs_refiner=callback_kwargs_refiner
    )
    end_gen_time = time.time()
    total_gen_time = end_gen_time - start_gen_time

    # --- 6. Save and Report ---
    output_filename = "sdxl_mue_phase2_image.png"
    generated_images[0].save(output_filename)
    
    print(f"\nImage saved to {output_filename}")
    print(f"Total Generation Time (excluding init): {total_gen_time:.2f} seconds")
    print("\nPhase 2 Test Complete. Check your terminal output for DIS Scores.")