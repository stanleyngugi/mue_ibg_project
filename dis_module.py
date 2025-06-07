# /workspace/mue_ibg_project/dis_module.py

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Any
# Import VaeImageProcessor here
from diffusers.image_processor import VaeImageProcessor


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

        # Initialize VaeImageProcessor here or before first use
        self._vae_image_processor = None # Will be initialized on first _decode_latents_to_pil call

        # --- Apply torch.compile to CLIP if enabled ---
        if compile_models:
            print("Compiling CLIP model for DIS with torch.compile...")
            self.clip_model.get_image_features = torch.compile(self.clip_model.get_image_features, mode="default", fullgraph=False)
            print("CLIP model for DIS compiled.")

        print("CLIP model loaded for DIS.")

    def _decode_latents_to_pil(self, latents: torch.Tensor, vae: Any) -> List[Image.Image]:
        """ 
        Decodes a batch of latents to PIL Images, processing them in smaller chunks
        to prevent CUDA Out of Memory errors.
        """
        decoded_images = []
        # Process one latent at a time to minimize memory usage during VAE decode
        decode_batch_size = 1 

        # Initialize VaeImageProcessor if it hasn't been already
        if self._vae_image_processor is None:
            self._vae_image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

        # Iterate over the latents in chunks
        for i in range(0, latents.shape[0], decode_batch_size):
            latent_chunk = latents[i : i + decode_batch_size]
            
            # Ensure the chunk is on the correct device and dtype for the VAE
            # This is important if `latents` came from CPU or a different dtype.
            latent_chunk = latent_chunk.to(vae.device, dtype=vae.dtype)

            # Decode the chunk with no_grad
            with torch.no_grad():
                decoded_latent_samples = vae.decode(latent_chunk / vae.config.scaling_factor).sample
            
            # Convert to PIL using the VaeImageProcessor
            pil_images_chunk = self._vae_image_processor.postprocess(decoded_latent_samples, output_type="pil")
            decoded_images.extend(pil_images_chunk)
            
            # Explicitly clear memory after each chunk. This can be very helpful.
            del latent_chunk, decoded_latent_samples, pil_images_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return decoded_images

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

        # original_pil will now be a list of 1 PIL Image
        original_pil = self._decode_latents_to_pil(latents, vae)
        original_inputs = self.clip_processor(images=original_pil, return_tensors="pt").to(self.device)
        original_embeds = F.normalize(self.clip_model.get_image_features(**original_inputs).float()) # Ensure float for F.normalize

        noise = torch.randn(
            (num_perturbations, *latents.shape[1:]),
            device=latents.device,
            dtype=latents.dtype
        ) * perturbation_scale

        perturbed_latents = latents + noise
        
        # perturbed_pils will now be a list of 'num_perturbations' PIL Images,
        # generated one by one in the _decode_latents_to_pil function.
        perturbed_pils = self._decode_latents_to_pil(perturbed_latents, vae)
        
        perturbed_inputs = self.clip_processor(images=perturbed_pils, return_tensors="pt").to(self.device)
        perturbed_embeds = F.normalize(self.clip_model.get_image_features(**perturbed_inputs).float()) # Ensure float for F.normalize

        similarity = F.cosine_similarity(original_embeds, perturbed_embeds, dim=-1)
        semantic_distance = (1.0 - similarity).mean().item()

        return semantic_distance