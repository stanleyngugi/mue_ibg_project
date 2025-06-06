# File: mue_pipeline.py (Corrected and Hardened)

import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline # Keep Img2Img for now, as it's common for refiner
from typing import Callable, List, Optional, Union, Dict, Any, Tuple

from dis_module import DISCalculator
from gloss_module import GlossCalculator

class MUEDiffusionPipeline(DiffusionPipeline):
    """
    A self-contained MUE pipeline with a manual denoising loop for adaptive guidance (DIS)
    and gradient-based steering (Gloss). Optimized with torch.compile.
    (Version 3.0 - Hardened against embedding errors and architectural flaws)
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        super().__init__()

        # --- Dynamic Device and Dtype Handling ---
        # Store the requested device and dtype temporarily
        _requested_device = device
        _requested_torch_dtype = torch_dtype

        # Determine the actual target device
        if _requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"WARNING: CUDA device '{_requested_device}' requested but CUDA is not available. Falling back to CPU.")
                self._target_device = "cpu"
            elif ":" in _requested_device:
                try:
                    device_id = int(_requested_device.split(":")[1])
                    if device_id >= torch.cuda.device_count():
                        print(f"WARNING: CUDA device '{_requested_device}' specified but only {torch.cuda.device_count()} CUDA devices found. Falling back to 'cuda:0'.")
                        self._target_device = "cuda:0"
                    else:
                        self._target_device = _requested_device
                except ValueError: # Handle cases like "cuda:abc"
                    print(f"WARNING: Invalid CUDA device string '{_requested_device}'. Falling back to 'cuda:0'.")
                    self._target_device = "cuda:0"
            else: # Just "cuda" implies "cuda:0"
                self._target_device = "cuda"
        else: # Explicitly CPU or other non-CUDA device
            self._target_device = _requested_device

        # Adjust dtype based on the final determined device
        if self._target_device == "cpu":
            if _requested_torch_dtype == torch.float16:
                print("WARNING: float16 is not recommended for CPU. Changing pipeline dtype to float32.")
            self.torch_dtype = torch.float32 # Force float32 for CPU
            if compile_models: # Disable compile on CPU
                print("WARNING: Disabling torch.compile as pipeline is running on CPU.")
                compile_models = False # torch.compile typically has minimal benefit and can be unstable on CPU
        else: # GPU
            self.torch_dtype = _requested_torch_dtype # Use the requested dtype for GPU

        # --- Model Loading Kwargs ---
        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        # Only specify the 'fp16' variant if we are actually using float16 on a CUDA device.
        # Otherwise, diffusers will load the default (usually fp32) or handle conversion.
        if self._target_device.startswith("cuda") and self.torch_dtype == torch.float16:
            model_loading_kwargs["variant"] = "fp16"
        else:
            model_loading_kwargs.pop("variant", None) # Remove 'variant' if not applicable

        print(f"Initializing MUEDiffusionPipeline with device: {self._target_device}, dtype: {self.torch_dtype}, compile_models: {compile_models}...")

        # --- Load Base components efficiently ---
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        del pipe_base

        # --- Load Refiner components efficiently ---
        # Keeping StableDiffusionXLImg2ImgPipeline here for consistency with typical diffusers refiner usage,
        # but ensure its encode_prompt is handled carefully.
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device) # Refiner only has text_encoder_2
        self.tokenizer_refiner = pipe_refiner.tokenizer_2 # Refiner only has tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner

        # --- Shared VAE ---
        # The 'fp16-fix' VAE is specifically FP16. If we're on CPU, we explicitly load it as FP32
        # to ensure compatibility and performance.
        vae_load_dtype = torch.float32 if self._target_device == "cpu" else self.torch_dtype
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=vae_load_dtype).to(self._target_device)

        # --- Model Compilation (if enabled and on GPU) ---
        if compile_models:
            print("Compiling models with torch.compile...")
            self.unet_base = torch.compile(self.unet_base, mode="default", fullgraph=True)
            self.unet_refiner = torch.compile(self.unet_refiner, mode="default", fullgraph=True)
            self.vae.decode = torch.compile(self.vae.decode, mode="default", fullgraph=True)
            self.text_encoder_base = torch.compile(self.text_encoder_base, mode="default", fullgraph=True)
            self.text_encoder_2_base = torch.compile(self.text_encoder_2_base, mode="default", fullgraph=True)
            self.text_encoder_refiner = torch.compile(self.text_encoder_refiner, mode="default", fullgraph=True)
            print("Models compiled.")

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_base_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Encodes prompts for the Base SDXL model. """
        temp_pipe = StableDiffusionXLPipeline(
            vae=self.vae, text_encoder=self.text_encoder_base, text_encoder_2=self.text_encoder_2_base,
            tokenizer=self.tokenizer_base, tokenizer_2=self.tokenizer_2_base, unet=self.unet_base,
            scheduler=self.scheduler_base
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt, device=self._target_device, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )
        del temp_pipe
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_refiner_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Encodes prompts for the SDXL Refiner model.
        Returns all 4 embedding tensors as produced by the Refiner's encode_prompt.
        """
        temp_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.vae, text_encoder=None, tokenizer=None, # Refiner only uses text_encoder_2
            text_encoder_2=self.text_encoder_refiner, tokenizer_2=self.tokenizer_refiner,
            unet=self.unet_refiner, scheduler=self.scheduler_refiner
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt, device=self._target_device, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )
        del temp_pipe
        # `prompt_embeds` will be (batch_size, seq_len, hidden_size) from text_encoder_2
        # `pooled_prompt_embeds` will be (batch_size, pooled_hidden_size) from text_encoder_2
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _prepare_added_cond_kwargs_base(self, height: int, width: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """ Prepares additional conditioning arguments for the Base UNet. """
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype)
        return {"add_time_ids": add_time_ids.repeat(batch_size, 1)}

    def _prepare_added_cond_kwargs_refiner(self, height: int, width: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """ Prepares additional conditioning arguments for the Refiner UNet. """
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype)
        add_aesthetic_embeds = torch.tensor([[2.5, 2.5, 2.5, 2.5, 2.5, 2.5]], device=self._target_device, dtype=self.torch_dtype)
        return {"add_time_ids": add_time_ids.repeat(batch_size, 1), "add_aesthetic_embeds": add_aesthetic_embeds.repeat(batch_size, 1)}

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps_base: int = 25,
        num_inference_steps_refiner: int = 5,
        denoising_end: float = 0.8,
        denoising_start: float = 0.8,
        initial_guidance_scale: float = 7.0,
        seed: int = 42,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_kwargs_base: Optional[Dict[str, Any]] = None,
        callback_on_step_end_kwargs_refiner: Optional[Dict[str, Any]] = None,
        gloss_calculator: Optional[GlossCalculator] = None,
        gloss_target: Optional[str] = None,
        gloss_strength: float = 0.0,
        gloss_gradient_clip_norm: float = 1.0,
        gloss_active_start_step: int = 0,
        output_type: str = "pil",
        **kwargs,
    ) -> Dict[str, Union[List[Image.Image], Any]]:
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if gloss_calculator:
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_base_prompt(prompt, negative_prompt)
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_refiner_prompt(prompt, negative_prompt)

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            generator=generator,
            device=self._target_device,
            dtype=self.vae.dtype
        )
        latents *= self.scheduler_base.init_noise_sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        full_prompt_embeds_b = torch.cat([neg_p_embeds_b, prompt_embeds_b])
        full_added_text_embeds_b = torch.cat([neg_pooled_embeds_b, pooled_embeds_b])
        added_cond_kwargs_b = self._prepare_added_cond_kwargs_base(height, width, batch_size)
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True)

            noise_pred = self.unet_base(
                latent_model_input, t,
                encoder_hidden_states=full_prompt_embeds_b,
                added_cond_kwargs={"text_embeds": full_added_text_embeds_b, **added_cond_kwargs_b}
            ).sample
            
            if is_gloss_active:
                 latents.requires_grad_(False)
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            if is_gloss_active:
                with torch.enable_grad():
                     latents.requires_grad_(True)
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latents, guided_noise_pred, t, self.scheduler_base, self.vae, gloss_strength, gloss_gradient_clip_norm)
                     latents.requires_grad_(False)
                latents = latents.detach() - gloss_grad
            
            latents = self.scheduler_base.step(guided_noise_pred, t, latents).prev_sample

            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        # 4. Refiner Denoising Loop
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self._target_device)
        timesteps_refiner = self.scheduler_refiner.timesteps[int(num_inference_steps_refiner * (1-denoising_start)):]
        
        # --- Crucial Fix for Refiner Embeddings ---
        # For the refiner, `prompt_embeds_r` (from text_encoder_2's last_hidden_state) is 3D: (batch, seq_len, 1280)
        # `pooled_embeds_r` (from text_encoder_2's pooled_output) is 2D: (batch, 1280)
        # The refiner's UNet `added_cond_kwargs['text_embeds']` expects a 2D tensor of shape (batch * 2, 2560).
        # This is formed by concatenating a pooled version of `prompt_embeds_r` with `pooled_embeds_r`.
        
        # We need to pool the 3D prompt_embeds_r down to 2D. A common method is taking the last token.
        # This is for the UNet's `added_cond_kwargs['text_embeds']` pathway.
        # The refiner also uses `encoder_hidden_states` as the 'long' embeddings.

        # Refiner's text conditioning logic (derived from diffusers SDXL refiner pipeline):
        # 1. Long embeddings go to `encoder_hidden_states` (the 3D ones)
        # 2. Combined short/pooled embeddings go to `added_cond_kwargs['text_embeds']`

        # Prepare `encoder_hidden_states` for the refiner UNet (3D tensor)
        full_encoder_hidden_states_r = torch.cat([neg_p_embeds_r, prompt_embeds_r])
        
        # Prepare `added_cond_kwargs['text_embeds']` for the refiner UNet (2D tensor from two pooled sources)
        # The first part is usually the *pooled* representation of the 3D prompt_embeds_r.
        # A common way to get a 2D pooled representation from a 3D tensor is to take the last token (EOS token)
        # or mean pooling.
        
        # Use a consistent pooling method for the refiner's `added_cond_kwargs['text_embeds']`
        # Here, we will use the `pooled_embeds_r` directly as the first part of the concatenation,
        # and typically, the `text_encoder_2` output is used for the main `encoder_hidden_states`.
        # The refiner UNet's `added_cond_kwargs["text_embeds"]` actually expects:
        # A concatenation of (batch_size, 1280) from `text_encoder_2`'s `pooled_output`
        # and another (batch_size, 1280) which is a specific projection of `text_encoder_2`'s `last_hidden_state`.
        # Diffusers pipelines often do this automatically.
        
        # Let's adjust based on the most common refiner UNet input structure for `added_cond_kwargs`:
        # It takes `text_embeds` which is concatenation of text_encoder_2's output and text_encoder_2's pooled output.
        # The `encode_prompt` method of StableDiffusionXLImg2ImgPipeline with text_encoder=None returns:
        # prompt_embeds (3D: batch, seq_len, hidden_size)
        # pooled_prompt_embeds (2D: batch, pooled_hidden_size)

        # The refiner UNet's `added_cond_kwargs` typically expects `text_embeds` as a 2D tensor
        # that is a concatenation of the *pooled* `prompt_embeds_r` and `pooled_embeds_r`.
        # A common way to pool the 3D `prompt_embeds_r` for this is to take the last token embedding.
        refiner_pooled_text_embeds = prompt_embeds_r[:, -1, :] # (batch, 1280) - last token's embedding
        refiner_pooled_text_embeds_neg = neg_p_embeds_r[:, -1, :] # (batch, 1280) - last token's embedding (negative)

        # Combine these with the actual `pooled_embeds_r` (which is already 2D)
        # This forms the (batch, 1280 + 1280) = (batch, 2560) tensor
        full_added_text_embeds_r_combined = torch.cat([refiner_pooled_text_embeds, pooled_embeds_r], dim=-1)
        full_neg_added_text_embeds_r_combined = torch.cat([refiner_pooled_text_embeds_neg, neg_pooled_embeds_r], dim=-1)

        # Concatenate for classifier-free guidance
        final_added_cond_text_embeds_r = torch.cat([full_neg_added_text_embeds_r_combined, full_added_text_embeds_r_combined])
        
        added_cond_kwargs_r = self._prepare_added_cond_kwargs_refiner(height, width, batch_size)
        
        # Combine all added conditioning into a single dictionary
        final_added_cond_kwargs_r = {"text_embeds": final_added_cond_text_embeds_r, **added_cond_kwargs_r}

        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # [IMPROVEMENT] Pass 3D prompt embeds via `encoder_hidden_states` and combined pooled embeds via `added_cond_kwargs`.
            noise_pred = self.unet_refiner(
                latent_model_input, t,
                encoder_hidden_states=full_encoder_hidden_states_r, # 3D tensor
                added_cond_kwargs=final_added_cond_kwargs_r # Contains the 2D text_embeds (2560 dim) + time_ids + aesthetic
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents).prev_sample
            
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i + base_denoise_end_step,
                    timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_refiner)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        # 5. VAE Decoding to Image
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return {"images": image}