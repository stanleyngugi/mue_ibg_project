import torch
import time
from mue_pipeline import MUEDiffusionPipeline
from typing import Dict, Any

# --- MUE Parameters ---
# For Adaptive Lambda (DIS)
MIN_GUIDANCE_SCALE = 4.0
MAX_GUIDANCE_SCALE = 12.0
DIS_STABLE_THRESHOLD = 0.002
DIS_UNSTABLE_THRESHOLD = 0.008
LAMBDA_INCREASE_RATE = 0.5
LAMBDA_DECREASE_RATE = 1.0

# For Gradient Steering (Gloss)
GLOSS_TARGET_CONCEPT = "sharp details, vibrant colors, clear focus, high contrast, photorealistic"
GLOSS_STRENGTH = 0.08  # Scaled down as our gradient is now more accurate
GLOSS_GRADIENT_CLIP_NORM = 0.5
GLOSS_ACTIVE_START_STEP = 5 # Start Gloss after the most chaotic initial steps

# --- 1. MUE Callback for DIS/Adaptive Lambda ---
def mue_callback(step: int, timestep: int, latents: torch.Tensor, current_guidance_scale: float, callback_kwargs: Dict[str, Any]) -> float:
    model_name = callback_kwargs.get("model_name", "Unknown Model")
    dis_calculator = callback_kwargs.get("dis_calculator")
    vae = callback_kwargs.get("vae")
    new_guidance_scale = current_guidance_scale

    if dis_calculator and vae:
        dis_score = dis_calculator.calculate_dis(latents=latents, vae=vae)
        print(f"[{model_name}] Step {step:02d} | DIS={dis_score:.5f} | In λ={current_guidance_scale:.2f}", end="")
        if dis_score < DIS_STABLE_THRESHOLD:
            new_guidance_scale += LAMBDA_INCREASE_RATE
            print(f" -> Stable, Out λ={new_guidance_scale:.2f}")
        elif dis_score > DIS_UNSTABLE_THRESHOLD:
            new_guidance_scale -= LAMBDA_DECREASE_RATE
            print(f" -> Unstable, Out λ={new_guidance_scale:.2f}")
        else:
            print(" -> Normal")
        return max(MIN_GUIDANCE_SCALE, min(new_guidance_scale, MAX_GUIDANCE_SCALE))
    return current_guidance_scale

# --- 2. Main Execution Block ---
if __name__ == "__main__":
    print("Starting Phase 4: Full MUE Test (Adaptive Lambda + Gloss)...")

    start_init_time = time.time()
    mue_pipe = MUEDiffusionPipeline(device="cuda" if torch.cuda.is_available() else "cpu")
    end_init_time = time.time()
    print(f"Pipeline initialization time: {end_init_time - start_init_time:.2f} seconds")

    # --- 3. Generation Parameters ---
    prompt = "photo of a majestic lion in the african savanna, sunset lighting"
    negative_prompt = "blurry, drawing, painting, cartoon, duplicate heads, deformed"
    seed = 2025

    # --- 4. Define Callback Kwargs ---
    callback_kwargs = {"model_name": "Base"} # Simplified for this test

    # --- 5. Perform Generation with Full MUE ---
    start_gen_time = time.time()
    result = mue_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        initial_guidance_scale=8.0,
        seed=seed,
        callback_on_step_end=mue_callback,
        callback_on_step_end_kwargs_base=callback_kwargs,
        # Gloss parameters
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