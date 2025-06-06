# File: gloss_module.py

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Any, Optional

class GlossCalculator:
    """
    Calculates the gradient for Look-Ahead Loss (Gloss) using CLIP embeddings.
    This gradient is used to steer latent generation towards a target concept.
    """
    def __init__(self, device: str = "cuda", compile_models: bool = False):
        self.device = device
        print(f"Loading CLIP model for Gloss calculation...")
        model_id = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.target_text_embedding = None

        if compile_models:
            print("Compiling CLIP model for Gloss with torch.compile...")
            self.clip_model.get_image_features = torch.compile(self.clip_model.get_image_features, mode="default", fullgraph=False)
            self.clip_model.get_text_features = torch.compile(self.clip_model.get_text_features, mode="default", fullgraph=False)
            print("CLIP model for Gloss compiled.")
        print(f"CLIP model loaded for Gloss.")

    def set_target(self, target_concept: str):
        """ Sets and pre-computes the embedding for the target concept. """
        if not target_concept:
            self.target_text_embedding = None
            print("Gloss target cleared.")
            return

        print(f"Gloss target set to: '{target_concept}'")
        with torch.no_grad():
            text_inputs = self.clip_processor(text=target_concept, return_tensors="pt").to(self.device)
            self.target_text_embedding = self.clip_model.get_text_features(**text_inputs).float()
            self.target_text_embedding = F.normalize(self.target_text_embedding, p=2, dim=-1)

    def _decode_latents_to_pil(self, latents: torch.Tensor, vae: Any) -> List[Image.Image]:
        """ Decodes latents to PIL Images for CLIP processing. """
        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).cpu().float().numpy()
        return [Image.fromarray((img * 255).astype("uint8")) for img in image]

    def calculate_gloss_gradient(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: int,
        scheduler: Any,
        vae: Any,
        gloss_strength: float,
        gradient_clip_norm: float
    ) -> torch.Tensor:
        """
        Calculates and returns the clipped and scaled Gloss gradient.
        """
        if self.target_text_embedding is None or gloss_strength == 0.0:
            return torch.zeros_like(latents)
        if not latents.requires_grad:
            raise ValueError("Latents must have requires_grad=True for Gloss calculation.")

        step_output = scheduler.step(noise_pred, current_timestep, latents)
        predicted_original_sample = step_output.pred_original_sample

        decoded_x0_pil = self._decode_latents_to_pil(predicted_original_sample, vae)
        clip_inputs = self.clip_processor(images=decoded_x0_pil, return_tensors="pt").to(self.device)
        image_embeds = F.normalize(self.clip_model.get_image_features(**clip_inputs))

        loss = (1.0 - F.cosine_similarity(image_embeds, self.target_text_embedding)).mean()
        gloss_grad = torch.autograd.grad(loss, latents, retain_graph=False)[0]
        torch.nn.utils.clip_grad_norm_([gloss_grad], gradient_clip_norm)

        return gloss_grad * gloss_strength