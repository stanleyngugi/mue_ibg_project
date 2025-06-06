# File: evaluation_utils.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import os
import time

# For FID, you'd typically install `cleanfid` (`pip install cleanfid`)
# and set up a path to a real image dataset. For this example, we'll
# focus on CLIP score and provide guidance for FID.
# from cleanfid import fid

def calculate_clip_score(
    images: List[Image.Image],
    prompts: List[str],
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    device: str
) -> float:
    """
    Calculates the average CLIP Score (cosine similarity between image and text embeddings).
    A higher score indicates better semantic alignment.
    """
    if not images or not prompts:
        return 0.0
    if len(images) != len(prompts):
        raise ValueError("Number of images must match number of prompts for CLIP score calculation.")

    # Ensure CLIP model is on the correct device and in eval mode
    clip_model.to(device).eval()

    total_similarity = 0.0
    batch_size = 8  # Process in batches to manage memory

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            # Encode images
            image_inputs = clip_processor(images=batch_images, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**image_inputs).float()
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Encode texts
            text_inputs = clip_processor(text=batch_prompts, padding=True, return_tensors="pt").to(device)
            text_features = clip_model.get_text_features(**text_inputs).float()
            text_features = F.normalize(text_features, p=2, dim=-1)

            # Calculate cosine similarity and sum
            similarity = F.cosine_similarity(image_features, text_features, dim=-1).sum().item()
            total_similarity += similarity

    return total_similarity / len(images)

def run_evaluation_batch(
    pipeline: Any,
    prompts: List[str],
    num_images_per_prompt: int,
    output_dir: str,
    config_name: str,
    **pipeline_kwargs  # All arguments for pipeline.__call__
) -> Tuple[List[Image.Image], List[str], float]:
    """
    Runs a batch of generations, saves images, and returns generated images/prompts for metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_generated_images = []
    all_corresponding_prompts = []

    print(f"\n--- Running evaluation for config: {config_name} ---")
    start_time_batch = time.time()

    total_image_count = len(prompts) * num_images_per_prompt
    image_num = 0

    for i, prompt in enumerate(prompts):
        print(f"Generating for prompt {i+1}/{len(prompts)}: '{prompt[:60]}...'")
        for j in range(num_images_per_prompt):
            image_num += 1
            # Pass a unique seed for each image within the batch for diversity and reproducibility
            current_seed = pipeline_kwargs.get("seed", 42) + (i * num_images_per_prompt) + j

            gen_start_time = time.time()
            # The pipeline is expected to return a dictionary with a list of images
            result = pipeline(
                prompt=prompt,
                seed=current_seed,
                **pipeline_kwargs
            )
            images = result['images']
            gen_end_time = time.time()
            print(f"  Image {image_num}/{total_image_count} generated in {gen_end_time - gen_start_time:.2f}s")

            # Save image
            img_path = os.path.join(output_dir, f"{config_name}_prompt{i:02d}_img{j:02d}.png")
            images[0].save(img_path)

            all_generated_images.extend(images)  # Assumes pipeline returns a list of images
            all_corresponding_prompts.extend([prompt] * len(images))

    end_time_batch = time.time()
    total_batch_time = end_time_batch - start_time_batch
    print(f"Total generation time for {config_name}: {total_batch_time:.2f} seconds ({len(all_generated_images)} images)")

    return all_generated_images, all_corresponding_prompts, total_batch_time

# How to calculate FID (requires setup)
# def get_fid_score(real_img_dir: str, generated_img_dir: str, device: str = "cuda") -> float:
#     """
#     Calculates FID score between real and generated image directories.
#     Requires pre-extracted real image statistics or the real image directory itself.
#     """
#     print(f"Calculating FID between '{real_img_dir}' and '{generated_img_dir}'...")
#     score = fid.fretchet_distance(real_img_dir, generated_img_dir, device=device)
#     return score