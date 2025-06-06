# File: mue_pipeline.py (Corrected and Hardened)

import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
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
        self._target_device = device
        self.torch_dtype = torch_dtype
        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype, "variant": "fp16",
            "use_safetensors": True, "low_cpu_mem_usage": True
        }

        print("Initializing MUEDiffusionPipeline with efficient component loading...")
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        del pipe_base

        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device)
        self.tokenizer_refiner = pipe_refiner.tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner

        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.torch_dtype).to(self._target_device)

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

    def _encode_base_prompt(self, prompt, negative_prompt):
        temp_pipe = StableDiffusionXLPipeline(
            vae=self.vae, text_encoder=self.text_encoder_base, text_encoder_2=self.text_encoder_2_base,
            tokenizer=self.tokenizer_base, tokenizer_2=self.tokenizer_2_base, unet=self.unet_base,
            scheduler=self.scheduler_base
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(prompt, self._target_device, 1, True, negative_prompt)
        del temp_pipe
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_refiner_prompt(self, prompt, negative_prompt):
        # [IMPROVEMENT] This function now correctly returns all 4 embedding tensors.
        # The faulty concatenation is removed, fixing the crash. The responsibility for
        # correctly using these embeddings is moved to the __call__ method.
        temp_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.vae, text_encoder=None, tokenizer=None,
            text_encoder_2=self.text_encoder_refiner, tokenizer_2=self.tokenizer_refiner,
            unet=self.unet_refiner, scheduler=self.scheduler_refiner
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(prompt, self._target_device, 1, True, negative_prompt)
        
        # [FIX] Do NOT concatenate here. Return the tensors as they are.
        del temp_pipe
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _prepare_added_cond_kwargs_base(self, height, width, batch_size):
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype)
        # [IMPROVEMENT] Removed redundant `* 1`
        return {"add_time_ids": add_time_ids.repeat(batch_size, 1)}

    def _prepare_added_cond_kwargs_refiner(self, height, width, batch_size):
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype)
        add_aesthetic_embeds = torch.tensor([[2.5, 2.5, 2.5, 2.5, 2.5, 2.5]], device=self._target_device, dtype=self.torch_dtype)
        return {"add_time_ids": add_time_ids.repeat(batch_size, 1), "add_aesthetic_embeds": add_aesthetic_embeds.repeat(batch_size, 1)}

    @torch.no_grad()
    def __call__(
        self, prompt, negative_prompt="", height=1024, width=1024,
        num_inference_steps_base=25, num_inference_steps_refiner=5,
        denoising_end=0.8, denoising_start=0.8, initial_guidance_scale=7.0,
        seed=42, callback_on_step_end=None,
        callback_on_step_end_kwargs_base=None, callback_on_step_end_kwargs_refiner=None,
        gloss_calculator=None, gloss_target=None, gloss_strength=0.0,
        gloss_gradient_clip_norm=1.0, gloss_active_start_step=0,
        output_type="pil", **kwargs
    ):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if gloss_calculator:
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_base_prompt(prompt, negative_prompt)
        
        # [FIX] Receive all 4 tensors from the corrected refiner encoding function.
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_refiner_prompt(prompt, negative_prompt)

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn((batch_size, 4, height // 8, width // 8), generator=generator, device=self._target_device, dtype=self.vae.dtype)
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
        
        # [FIX] Correctly prepare all embeddings for the refiner UNet.
        # This resolves the latent bug where pooled embeddings were ignored.
        full_prompt_embeds_r = torch.cat([neg_p_embeds_r, prompt_embeds_r])
        full_added_text_embeds_r = torch.cat([neg_pooled_embeds_r, pooled_embeds_r])
        added_cond_kwargs_r = self._prepare_added_cond_kwargs_refiner(height, width, batch_size)
        
        # Combine all added conditioning into a single dictionary
        final_added_cond_kwargs_r = {"text_embeds": full_added_text_embeds_r, **added_cond_kwargs_r}

        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # [IMPROVEMENT] Call the UNet using the standard, robust pattern.
            # Pass prompt embeds via `encoder_hidden_states` and the rest via `added_cond_kwargs`.
            noise_pred = self.unet_refiner(
                latent_model_input, t,
                encoder_hidden_states=full_prompt_embeds_r,
                added_cond_kwargs=final_added_cond_kwargs_r
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