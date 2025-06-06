# File: mue_pipeline.py

import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from typing import Callable, List, Optional, Union, Dict, Any

from dis_module import DISCalculator
from gloss_module import GlossCalculator

class MUEDiffusionPipeline(DiffusionPipeline):
    """
    A self-contained MUE pipeline with a manual denoising loop for adaptive guidance (DIS)
    and gradient-based steering (Gloss). Optimized with torch.compile.
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        model_loading_kwargs = {"torch_dtype": self.torch_dtype, "variant": "fp16", "use_safetensors": True}

        print("Initializing MUEDiffusionPipeline with efficient component loading...")
        # Load Base components efficiently
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self.device)
        self.text_encoder_base = pipe_base.text_encoder.to(self.device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self.device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        del pipe_base

        # Load Refiner components efficiently
        pipe_refiner = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self.device)
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self.device)
        self.tokenizer_refiner = pipe_refiner.tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner

        # Shared VAE
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.torch_dtype).to(self.device)

        if compile_models:
            print("Compiling UNets and VAE with torch.compile...")
            self.unet_base = torch.compile(self.unet_base, mode="default", fullgraph=True)
            self.unet_refiner = torch.compile(self.unet_refiner, mode="default", fullgraph=True)
            self.vae.decode = torch.compile(self.vae.decode, mode="default", fullgraph=True)
            print("UNets and VAE compiled.")

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_prompt(self, text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, negative_prompt):
        # Create a temporary minimal pipeline for prompt encoding
        temp_pipe = StableDiffusionXLPipeline(
            vae=self.vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer,
            tokenizer_2=tokenizer_2, unet=self.unet_base, scheduler=self.scheduler_base
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
        del temp_pipe
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _prepare_added_cond_kwargs(self, height, width, batch_size):
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self.device, dtype=self.torch_dtype)
        return {"add_time_ids": add_time_ids.repeat(batch_size, 1)}

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = "",
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
        **kwargs,
    ) -> Dict[str, List[Image.Image]]:
        batch_size = 1
        if gloss_calculator:
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_prompt(
            self.text_encoder_base, self.text_encoder_2_base, self.tokenizer_base, self.tokenizer_2_base, prompt, negative_prompt)
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_prompt(
            self.text_encoder_base, self.text_encoder_refiner, self.tokenizer_base, self.tokenizer_refiner, prompt, negative_prompt)

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((batch_size, 4, height // 8, width // 8), generator=generator, device=self.device, dtype=self.vae.dtype)
        latents = latents * self.scheduler_base.init_noise_sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self.device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        added_cond_kwargs = self._prepare_added_cond_kwargs(height, width, batch_size)
        prompt_embeds_b = torch.cat([neg_p_embeds_b, prompt_embeds_b])
        added_text_embeds_b = torch.cat([neg_pooled_embeds_b, pooled_embeds_b])
        
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            # Enable gradient tracking for Gloss if needed
            if gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0:
                latent_model_input.requires_grad_(True)

            noise_pred = self.unet_base(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds_b,
                added_cond_kwargs={**added_cond_kwargs, "text_embeds": added_text_embeds_b}
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- Gloss Gradient Application ---
            if gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0:
                with torch.enable_grad():
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latent_model_input[:batch_size], guided_noise_pred, t, self.scheduler_base, self.vae, gloss_strength, gloss_gradient_clip_norm)
                latents = latents - gloss_grad
                latents = latents.detach() # Detach after steering

            # --- DIS Callback for Adaptive Lambda ---
            if callback_on_step_end:
                current_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)

            latents = self.scheduler_base.step(guided_noise_pred, t, latents).prev_sample

        # 4. Refiner Denoising Loop
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self.device)
        timesteps_refiner = self.scheduler_refiner.timesteps[int(num_inference_steps_refiner * (1-denoising_start)):]
        prompt_embeds_r = torch.cat([neg_p_embeds_r, prompt_embeds_r])
        added_text_embeds_r = torch.cat([neg_pooled_embeds_r, pooled_embeds_r])

        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            noise_pred = self.unet_refiner(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds_r,
                added_cond_kwargs={**added_cond_kwargs, "text_embeds": added_text_embeds_r}
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents).prev_sample

        # 5. VAE Decoding
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = self.numpy_to_pil((image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy())
        
        return {"images": image}