# File: run_phase5.py

import torch
import time
import os
from collections import defaultdict

from mue_pipeline import MUEDiffusionPipeline
from dis_module import DISCalculator
from gloss_module import GlossCalculator
from evaluation_utils import calculate_clip_score, run_evaluation_batch
from transformers import CLIPProcessor, CLIPModel

# --- MUE Parameters for Adaptive Lambda ---
MIN_GUIDANCE_SCALE = 4.0
MAX_GUIDANCE_SCALE = 12.0
DIS_STABLE_THRESHOLD = 0.002
DIS_UNSTABLE_THRESHOLD = 0.008
LAMBDA_INCREASE_RATE = 0.5
LAMBDA_DECREASE_RATE = 1.0

# --- MUE Parameters for Gloss ---
GLOSS_TARGET_CONCEPT = "sharp details, vibrant colors, clear focus, high contrast, golden hour lighting"
GLOSS_STRENGTH = 0.4
GLOSS_GRADIENT_CLIP_NORM = 1.0
GLOSS_ACTIVE_START_STEP = 4

# --- Evaluation Parameters ---
EVAL_PROMPTS = [
    "photorealistic image of a lone astronaut exploring a vibrant alien jungle, bioluminescent plants, misty atmosphere",
    "a highly detailed hyperrealistic portrait of an elderly wizard reading an ancient glowing spellbook in a dimly lit, cozy library, dust motes in the air, volumetric lighting",
    "a vibrant, detailed portrait of a mystical forest spirit with intricate glowing patterns on its skin, surrounded by luminous flora, deep focus, sharp edges"
]
NUM_IMAGES_PER_PROMPT = 2 # Keep this low for faster testing
OUTPUT_BASE_DIR = "mue_eval_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MUE callback function ---
def mue_callback(step: int, timestep: int, latents: torch.Tensor, current_guidance_scale: float, callback_kwargs: Dict[str, Any]) -> float:
    dis_calculator: DISCalculator = callback_kwargs.get("dis_calculator_instance")
    vae = callback_kwargs.get("vae")

    if not dis_calculator or not vae:
        return current_guidance_scale

    dis_score = dis_calculator.calculate_dis(
        latents=latents, vae=vae, perturbation_scale=0.01, num_perturbations=5
    )

    if dis_score < DIS_STABLE_THRESHOLD:
        new_guidance_scale = current_guidance_scale + LAMBDA_INCREASE_RATE
    elif dis_score > DIS_UNSTABLE_THRESHOLD:
        new_guidance_scale = current_guidance_scale - LAMBDA_DECREASE_RATE
    else:
        new_guidance_scale = current_guidance_scale

    return max(MIN_GUIDANCE_SCALE, min(new_guidance_scale, MAX_GUIDANCE_SCALE))

# --- Main Evaluation Loop ---
if __name__ == "__main__":
    print("Starting Phase 5: Evaluation and Optimization Test...")

    configurations = {
        "Baseline": {"compile_models": False, "initial_guidance_scale": 7.0},
        "Baseline_Compiled": {"compile_models": True, "initial_guidance_scale": 7.0},
        "MUE_Full": {"compile_models": False, "initial_guidance_scale": 7.0, "use_mue": True},
        "MUE_Full_Compiled": {"compile_models": True, "initial_guidance_scale": 7.0, "use_mue": True},
    }
    results = defaultdict(dict)

    # Initialize a global CLIP model for metrics calculation
    print("Initializing global CLIP model for metrics...")
    global_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    global_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    global_clip_model.get_image_features = torch.compile(global_clip_model.get_image_features, mode="default", fullgraph=False)
    print("Global CLIP model compiled for metrics.")

    for config_name, config_params in configurations.items():
        print(f"\n===== Running Configuration: {config_name} =====")

        mue_pipe = MUEDiffusionPipeline(
            device=DEVICE, torch_dtype=torch.float16, compile_models=config_params["compile_models"]
        )

        pipeline_kwargs = {
            "negative_prompt": "blurry, low quality, distorted, bad art, cartoon, anime, ugly, deformed, text, oversaturated",
            "height": 1024, "width": 1024,
            "num_inference_steps_base": 25, "num_inference_steps_refiner": 5,
            "denoising_end": 0.8, "denoising_start": 0.8,
            "initial_guidance_scale": config_params["initial_guidance_scale"],
        }

        if config_params.get("use_mue", False):
            dis_calculator = DISCalculator(device=DEVICE, compile_models=config_params["compile_models"])
            gloss_calculator = GlossCalculator(device=DEVICE, compile_models=config_params["compile_models"])
            pipeline_kwargs.update({
                "callback_on_step_end": mue_callback,
                "callback_on_step_end_kwargs_base": {"dis_calculator_instance": dis_calculator},
                "gloss_calculator": gloss_calculator,
                "gloss_target": GLOSS_TARGET_CONCEPT,
                "gloss_strength": GLOSS_STRENGTH,
                "gloss_gradient_clip_norm": GLOSS_GRADIENT_CLIP_NORM,
                "gloss_active_start_step": GLOSS_ACTIVE_START_STEP,
            })

        generated_images, corresponding_prompts, total_gen_time = run_evaluation_batch(
            pipeline=mue_pipe, prompts=EVAL_PROMPTS, num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
            output_dir=os.path.join(OUTPUT_BASE_DIR, config_name), config_name=config_name,
            **pipeline_kwargs
        )

        clip_score = calculate_clip_score(
            generated_images, corresponding_prompts, global_clip_processor, global_clip_model, DEVICE
        )

        results[config_name]["Total Gen Time (s)"] = total_gen_time
        results[config_name]["Avg Time per Image (s)"] = total_gen_time / len(generated_images)
        results[config_name]["CLIP Score"] = clip_score

        print(f"Metrics for {config_name}: CLIP Score={clip_score:.4f}, Avg Time/Img={results[config_name]['Avg Time per Image (s)']:.2f}s")

        del mue_pipe
        if "dis_calculator" in locals(): del dis_calculator
        if "gloss_calculator" in locals(): del gloss_calculator
        torch.cuda.empty_cache()
        time.sleep(2)

    print("\n\n===== Evaluation Summary =====")
    for config_name, metrics in results.items():
        print(f"\n--- {config_name} ---")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\nPhase 5 Test Complete. Review the results in the console and the generated images in the 'mue_eval_results' directory.")