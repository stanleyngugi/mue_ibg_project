# File: dis_module.py

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Any

class DISCalculator:
    """
    Calculates the Diffusion Instability Signal (DIS) using a batch-optimized approach.
    GU is approximated by semantic sensitivity to small latent perturbations using CLIP embeddings.
    """
    def __init__(self, device: str = "cuda", compile_models: bool = False):
        self.device = device
        print("Loading CLIP model for DIS calculation...")
        model_id = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

        # --- Apply torch.compile to CLIP if enabled ---
        if compile_models:
            print("Compiling CLIP model for DIS with torch.compile...")
            self.clip_model.get_image_features = torch.compile(self.clip_model.get_image_features, mode="default", fullgraph=False)
            print("CLIP model for DIS compiled.")

        print("CLIP model loaded for DIS.")

    def _decode_latents_to_pil(self, latents: torch.Tensor, vae: Any) -> List[Image.Image]:
        """ Decodes latents to PIL Images for CLIP processing. """
        latents = latents.to(vae.device, dtype=vae.dtype)
        # The VAE decode function handles the scaling internally
        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).cpu().float().numpy()
        return [Image.fromarray((img * 255).astype("uint8")) for img in image]

    @torch.no_grad()
    def calculate_dis(
        self,
        latents: torch.Tensor,
        vae: Any,
        perturbation_scale: float = 0.005,
        num_perturbations: int = 5,
    ) -> float:
        """
        Calculates the DIS score using a highly efficient batch-processing method.
        """
        if num_perturbations == 0:
            return 0.0

        original_pil = self._decode_latents_to_pil(latents, vae)
        original_inputs = self.clip_processor(images=original_pil, return_tensors="pt").to(self.device)
        original_embeds = F.normalize(self.clip_model.get_image_features(**original_inputs))

        noise = torch.randn(
            (num_perturbations, *latents.shape[1:]),
            device=latents.device,
            dtype=latents.dtype
        ) * perturbation_scale

        perturbed_latents = latents + noise
        perturbed_pils = self._decode_latents_to_pil(perturbed_latents, vae)
        perturbed_inputs = self.clip_processor(images=perturbed_pils, return_tensors="pt").to(self.device)
        perturbed_embeds = F.normalize(self.clip_model.get_image_features(**perturbed_inputs))

        similarity = F.cosine_similarity(original_embeds, perturbed_embeds, dim=-1)
        semantic_distance = (1.0 - similarity).mean().item()

        return semantic_distance