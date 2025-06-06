import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from typing import Callable, List, Optional, Union, Dict, Any, Tuple

from dis_module import DISCalculator # Assuming these modules exist and are correct
from gloss_module import GlossCalculator # Assuming these modules exist and are correct

class MUEDiffusionPipeline(DiffusionPipeline):
    """
    A self-contained MUE pipeline with a manual denoising loop for adaptive guidance (DIS)
    and gradient-based steering (Gloss). Optimized with torch.compile.
    (Version 2.0 - With critical bug fixes and enhancements)
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        # Initialize the base DiffusionPipeline class first.
        # This sets up core pipeline functionalities like self.image_processor, etc.
        super().__init__()

        # Use a private attribute for the target device to avoid clashes with
        # DiffusionPipeline's internal 'device' property management.
        self._target_device = device
        self.torch_dtype = torch_dtype

        # Common kwargs for loading models
        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype,
            "variant": "fp16",
            "use_safetensors": True,
            "low_cpu_mem_usage": True # Good practice for VRAM efficiency during loading
        }

        print("Initializing MUEDiffusionPipeline with efficient component loading...")

        # --- Load Base components efficiently ---
        # We load the full pipeline temporarily to extract individual components,
        # ensuring they are correctly configured and then moved to the target device.
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        
        # Assign components to self and move to target device
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        
        # VAE is loaded separately to ensure consistency and fp16 fix
        # Ensure VAE is also explicitly moved to target device if it's not done later
        # It's defined as shared, so we'll load it once below.

        del pipe_base # Free up memory from the temporary pipeline object

        # --- Load Refiner components efficiently ---
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        # Refiner model only has text_encoder_2 and tokenizer_2
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device)
        self.tokenizer_refiner = pipe_refiner.tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner

        # --- Shared VAE ---
        # Load the VAE from the fp16 fix, ensuring it's moved to the target device
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.torch_dtype).to(self._target_device)

        # --- Model Compilation (if enabled) ---
        if compile_models:
            print("Compiling UNets, VAE, and Text Encoders with torch.compile...")
            self.unet_base = torch.compile(self.unet_base, mode="default", fullgraph=True)
            self.unet_refiner = torch.compile(self.unet_refiner, mode="default", fullgraph=True)
            self.vae.decode = torch.compile(self.vae.decode, mode="default", fullgraph=True)
            self.text_encoder_base = torch.compile(self.text_encoder_base, mode="default", fullgraph=True)
            self.text_encoder_2_base = torch.compile(self.text_encoder_2_base, mode="default", fullgraph=True)
            self.text_encoder_refiner = torch.compile(self.text_encoder_refiner, mode="default", fullgraph=True)
            print("Models compiled.")

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_base_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Encodes prompts for the Base SDXL model using a temporary pipeline.
        This leverages `diffusers`'s robust `encode_prompt` method.
        """
        # Create a temporary pipeline instance using the pre-loaded components.
        # This is safe as the components are already on the correct device.
        temp_pipe = StableDiffusionXLPipeline(
            vae=self.vae, 
            text_encoder=self.text_encoder_base, 
            text_encoder_2=self.text_encoder_2_base,
            tokenizer=self.tokenizer_base, 
            tokenizer_2=self.tokenizer_2_base, 
            unet=self.unet_base,
            scheduler=self.scheduler_base
        )
        
        # Call the encode_prompt method with the correct device.
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt=prompt, 
                device=self._target_device, 
                num_images_per_prompt=1, 
                do_classifier_free_guidance=True, 
                negative_prompt=negative_prompt
            )
        
        # Clean up the temporary pipeline object to free memory.
        del temp_pipe
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_refiner_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Encodes prompts for the SDXL Refiner model.
        Crucially, passes `None` for `text_encoder` and `tokenizer` as the refiner model
        only uses `text_encoder_2` and `tokenizer_2`.
        """
        # Create a temporary pipeline instance for the refiner.
        # This pipeline's __init__ expects 'text_encoder' and 'tokenizer'
        # even if the refiner model itself doesn't use them.
        # Pass None for these to satisfy the signature.
        temp_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.vae, 
            text_encoder=None,  # Crucial fix: Refiner model does not use text_encoder (text_encoder_1)
            tokenizer=None,     # Crucial fix: Refiner model does not use tokenizer (tokenizer_1)
            text_encoder_2=self.text_encoder_refiner,
            tokenizer_2=self.tokenizer_refiner,
            unet=self.unet_refiner,
            scheduler=self.scheduler_refiner
        )
        
        # Call the encode_prompt method, ensuring the correct device is used.
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt=prompt, 
                device=self._target_device, 
                num_images_per_prompt=1, 
                do_classifier_free_guidance=True, 
                negative_prompt=negative_prompt
            )
        
        # Concatenate the main prompt embeds with the pooled embeds for the refiner.
        # This is how SDXL handles the conditioning for the refiner.
        prompt_embeds_combined = torch.cat([prompt_embeds, pooled_prompt_embeds], dim=-1)
        negative_prompt_embeds_combined = torch.cat([negative_prompt_embeds, negative_pooled_prompt_embeds], dim=-1)
        
        # Clean up the temporary pipeline object.
        del temp_pipe
        return prompt_embeds_combined, negative_prompt_embeds_combined

    def _prepare_added_cond_kwargs_base(self, height: int, width: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """ Prepares additional conditioning arguments for the Base UNet. """
        # `add_time_ids` are required for SDXL base.
        add_time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]], 
            device=self._target_device, 
            dtype=self.torch_dtype
        )
        return {"add_time_ids": add_time_ids.repeat(batch_size * 1, 1)}

    def _prepare_added_cond_kwargs_refiner(self, height: int, width: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """ Prepares additional conditioning arguments for the Refiner UNet. """
        # Refiner uses `add_time_ids` and `add_aesthetic_embeds`.
        add_time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]], 
            device=self._target_device, 
            dtype=self.torch_dtype
        )
        # Standard aesthetic embedding for refiner.
        add_aesthetic_embeds = torch.tensor(
            [[2.5, 2.5, 2.5, 2.5, 2.5, 2.5]], 
            device=self._target_device, 
            dtype=self.torch_dtype
        )
        
        return {
            "add_time_ids": add_time_ids.repeat(batch_size, 1), 
            "add_aesthetic_embeds": add_aesthetic_embeds.repeat(batch_size, 1)
        }

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
        denoising_start: float = 0.8, # Typically 0.8 for refiner, but allows custom start
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
        output_type: str = "pil", # Add output_type to control return format
        **kwargs,
    ) -> Dict[str, Union[List[Image.Image], Any]]:
        """
        Main call method for the MUE Diffusion Pipeline.
        Generates an image using a two-stage process (Base + Refiner) with custom logic.
        """
        if isinstance(prompt, str):
            batch_size = 1
        else: # Assumes prompt is a list of strings
            batch_size = len(prompt)

        # Initialize GlossCalculator target if provided
        if gloss_calculator:
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        # Encode prompts for the Base model
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = \
            self._encode_base_prompt(prompt, negative_prompt)
        
        # Encode prompts for the Refiner model (with the crucial fix)
        prompt_embeds_r, neg_p_embeds_r = self._encode_refiner_prompt(prompt, negative_prompt)

        # 2. Latent & Timestep Preparation for Base model
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8), 
            generator=generator, 
            device=self._target_device, 
            dtype=self.vae.dtype
        )
        latents = latents * self.scheduler_base.init_noise_sigma # Scale latents for DPM-Solver++

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        # Prepare combined embeddings and added conditioning for the Base UNet
        # Concatenate negative and positive prompt embeds for classifier-free guidance
        full_prompt_embeds_b = torch.cat([neg_p_embeds_b, prompt_embeds_b])
        full_added_text_embeds_b = torch.cat([neg_pooled_embeds_b, pooled_embeds_b])
        added_cond_kwargs_b = self._prepare_added_cond_kwargs_base(height, width, batch_size)
        
        # Prepare callback kwargs for the base stage
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            # Gloss integration: Enable gradient tracking if Gloss is active for this step
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True)

            # Predict noise residual
            noise_pred = self.unet_base(
                latent_model_input, t,
                encoder_hidden_states=full_prompt_embeds_b,
                added_cond_kwargs={"text_embeds": full_added_text_embeds_b, **added_cond_kwargs_b}
            ).sample
            
            # Disable gradient tracking if it was enabled for Gloss
            if is_gloss_active:
                 latents.requires_grad_(False)
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- Gloss Gradient Application ---
            if is_gloss_active:
                with torch.enable_grad(): # Ensure gradients are enabled for this computation
                     latents.requires_grad_(True) # Re-enable if disabled by detach or previous step
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latents, guided_noise_pred, t, self.scheduler_base, self.vae, gloss_strength, gloss_gradient_clip_norm)
                     latents.requires_grad_(False) # Disable immediately after use
                
                # Apply gradient descent to latents
                latents = latents.detach() - gloss_grad
            
            # --- Scheduler Step (Denoise) ---
            latents = self.scheduler_base.step(guided_noise_pred, t, latents).prev_sample

            # --- Callback for Adaptive Guidance (e.g., DIS) ---
            if callback_on_step_end:
                # The callback can optionally return a new guidance scale
                new_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)
                current_guidance_scale = new_guidance_scale or current_guidance_scale # Update if callback returned a new value


        # 4. Refiner Denoising Loop
        # Set timesteps for the refiner scheduler
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self._target_device)
        # Determine the refiner's starting timestep based on denoising_start
        timesteps_refiner = self.scheduler_refiner.timesteps[int(num_inference_steps_refiner * (1-denoising_start)):]
        
        # Prepare combined embeddings and added conditioning for the Refiner UNet
        full_prompt_embeds_r = torch.cat([neg_p_embeds_r, prompt_embeds_r])
        added_cond_kwargs_r = self._prepare_added_cond_kwargs_refiner(height, width, batch_size)
        
        # Prepare callback kwargs for the refiner stage
        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        # The refiner loop does not typically involve Gloss, as it's the final stage.
        # If Gloss is desired for the refiner, additional logic would be needed here.
        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # Refiner's added_cond_kwargs structure
            # Note: Refiner typically uses its combined prompt_embeds directly in added_cond_kwargs.text_embeds
            added_cond_kwargs = {"text_embeds": full_prompt_embeds_r, **added_cond_kwargs_r}

            # Predict noise residual
            noise_pred = self.unet_refiner(
                latent_model_input, t,
                encoder_hidden_states=None, # Refiner typically doesn't use primary encoder_hidden_states here
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler Step (Denoise)
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents).prev_sample
            
            # Callback for Adaptive Guidance in Refiner stage
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i + base_denoise_end_step, # Adjust step count to be continuous
                    timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_refiner)
                current_guidance_scale = new_guidance_scale or current_guidance_scale


        # 5. VAE Decoding to Image
        # Decode the final latents back into a pixel image.
        # Ensure latents are scaled correctly before decoding.
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # Post-process the image (e.g., scale to [0, 255] and convert to PIL format).
        # DiffusionPipeline has an `image_processor` attribute initialized by `super().__init__()`.
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return {"images": image}