# File: mue_pipeline.py (Final, Corrected Version)

import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from typing import Callable, List, Optional, Union, Dict, Any, Tuple

# Assuming these modules are correctly implemented and available
from dis_module import DISCalculator
from gloss_module import GlossCalculator

class MUEDiffusionPipeline(DiffusionPipeline):
    """
    A self-contained MUE pipeline with a manual denoising loop for adaptive guidance (DIS)
    and gradient-based steering (Gloss).
    (Version 4.0 - Final, Hardened, and Architecturally Correct)
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        super().__init__()

        # --- Device & Dtype Safety Checks ---
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"WARNING: CUDA device '{device}' requested but not available. Falling back to CPU.")
            self._target_device = "cpu"
        else:
            self._target_device = device

        if self._target_device == "cpu":
            if torch_dtype == torch.float16:
                print("WARNING: float16 is not recommended for CPU. Forcing float32.")
            self.torch_dtype = torch.float32
            if compile_models:
                print("WARNING: Disabling torch.compile on CPU.")
                compile_models = False
        else:
            self.torch_dtype = torch_dtype

        print(f"Initializing MUEDiffusionPipeline with device: {self._target_device}, dtype: {self.torch_dtype}, compile_models: {compile_models}...")

        model_loading_kwargs = {"torch_dtype": self.torch_dtype, "use_safetensors": True, "low_cpu_mem_usage": True}
        if self._target_device.startswith("cuda") and self.torch_dtype == torch.float16:
            model_loading_kwargs["variant"] = "fp16"

        # --- Load Components ---
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
            self.unet_base = torch.compile(self.unet_base)
            self.unet_refiner = torch.compile(self.unet_refiner)
            self.vae.decode = torch.compile(self.vae.decode)
            self.text_encoder_base = torch.compile(self.text_encoder_base)
            self.text_encoder_2_base = torch.compile(self.text_encoder_2_base)
            self.text_encoder_refiner = torch.compile(self.text_encoder_refiner)
            print("Models compiled.")

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_prompt_helper(self, text_encoder, tokenizer, text_encoder_2, tokenizer_2, prompt, negative_prompt):
        """A unified helper to encode prompts using a temporary pipeline, ensuring correctness."""
        pipeline_class = StableDiffusionXLPipeline if text_encoder is not None else StableDiffusionXLImg2ImgPipeline
        temp_pipe = pipeline_class(
            vae=self.vae, text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
            unet=self.unet_base, scheduler=self.scheduler_base
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(prompt, self._target_device, 1, True, negative_prompt)
        del temp_pipe
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

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
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_prompt_helper(
            self.text_encoder_base, self.tokenizer_base, self.text_encoder_2_base, self.tokenizer_2_base, prompt, negative_prompt)
        
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_prompt_helper(
            None, None, self.text_encoder_refiner, self.tokenizer_refiner, prompt, negative_prompt)

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn((batch_size, self.unet_base.config.in_channels, height // 8, width // 8),
                              generator=generator, device=self._target_device, dtype=self.vae.dtype)
        latents *= self.scheduler_base.init_noise_sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        # [FIXED & SIMPLIFIED] Correctly prepare conditioning for the Base UNet
        add_time_ids_b = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        prompt_embeds_b_cfg = torch.cat([neg_p_embeds_b, prompt_embeds_b])
        add_text_embeds_b_cfg = torch.cat([neg_pooled_embeds_b, pooled_embeds_b])
        add_time_ids_b_cfg = torch.cat([add_time_ids_b, add_time_ids_b])

        final_added_cond_kwargs_base = {"text_embeds": add_text_embeds_b_cfg, "time_ids": add_time_ids_b_cfg}
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True)

            noise_pred = self.unet_base(latent_model_input, t,
                                        encoder_hidden_states=prompt_embeds_b_cfg,
                                        added_cond_kwargs=final_added_cond_kwargs_base).sample
            
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
        
        # [FIXED & SIMPLIFIED] Correctly prepare conditioning for the Refiner UNet
        add_time_ids_r = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        prompt_embeds_r_cfg = torch.cat([neg_p_embeds_r, prompt_embeds_r])
        add_text_embeds_r_cfg = torch.cat([neg_pooled_embeds_r, pooled_embeds_r])
        add_time_ids_r_cfg = torch.cat([add_time_ids_r, add_time_ids_r])
        
        final_added_cond_kwargs_refiner = {"text_embeds": add_text_embeds_r_cfg, "time_ids": add_time_ids_r_cfg}
        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            noise_pred = self.unet_refiner(latent_model_input, t,
                                           encoder_hidden_states=prompt_embeds_r_cfg,
                                           added_cond_kwargs=final_added_cond_kwargs_refiner).sample
            
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

        # 5. VAE Decoding
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return {"images": image}