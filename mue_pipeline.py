import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from typing import Callable, List, Optional, Union, Dict, Any, Tuple
import inspect # For checking default parameters of scheduler's step method

# Assuming these modules are correctly implemented and available
# from dis_module import DISCalculator # Uncomment if you have this
# from gloss_module import GlossCalculator # Uncomment if you have this

class MUEDiffusionPipeline(DiffusionPipeline):
    """
    A self-contained MUE pipeline with a manual denoising loop for adaptive guidance (DIS)
    and gradient-based steering (Gloss).
    (Version 6.2 - Definitive fix for Refiner added_cond_kwargs dimensional mismatch.
    The fix targets `unet_refiner.config.addition_embed_type` to align input dimensions.)
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        """
        Initializes the MUEDiffusionPipeline.

        Args:
            device (str): The device to run the pipeline on (e.g., "cuda", "cuda:0", "cpu").
            torch_dtype (torch.dtype): The torch data type to use for models (e.g., torch.float16, torch.float32).
            compile_models (bool): Whether to compile models with torch.compile for potential speedup (CUDA only).
        """
        super().__init__()

        # --- Device & Dtype Safety Checks ---
        _requested_device = device
        _requested_torch_dtype = torch_dtype

        if _requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"WARNING: CUDA device '{_requested_device}' requested but CUDA is not available. Falling back to CPU.")
                self._target_device = "cpu"
            else:
                try:
                    if ":" in _requested_device:
                        device_id_str = _requested_device.split(":")[1]
                        if not device_id_str.isdigit():
                            raise ValueError(f"Invalid CUDA device ID: {device_id_str}")
                        device_id = int(device_id_str)
                        if device_id >= torch.cuda.device_count():
                            print(f"WARNING: CUDA device '{_requested_device}' specified but only {torch.cuda.device_count()} CUDA devices found. Falling back to 'cuda:0'.")
                            self._target_device = "cuda:0"
                        else:
                            self._target_device = _requested_device
                    else:
                        self._target_device = "cuda:0"
                except ValueError as e:
                    print(f"WARNING: Invalid CUDA device string '{_requested_device}' ({e}). Falling back to 'cuda:0'.")
                    self._target_device = "cuda:0"
        else:
            self._target_device = _requested_device

        if self._target_device == "cpu":
            if _requested_torch_dtype == torch.float16:
                print("WARNING: float16 is not recommended for CPU. Changing pipeline dtype to float32.")
            self.torch_dtype = torch.float32
            if compile_models:
                print("WARNING: Disabling torch.compile as pipeline is running on CPU (minimal performance benefit/potential instability).")
                compile_models = False
        else:
            self.torch_dtype = _requested_torch_dtype

        print(f"Initializing MUEDiffusionPipeline with device: {self._target_device}, dtype: {self.torch_dtype}, compile_models: {compile_models}...")

        # Common kwargs for model loading
        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        if self._target_device.startswith("cuda") and self.torch_dtype == torch.float16:
            model_loading_kwargs["variant"] = "fp16"
        else:
            model_loading_kwargs.pop("variant", None) # Ensure variant is not set if not fp16

        # --- Load Components Efficiently ---
        print("Loading pipeline components...")

        # Load Base Pipeline and extract components
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)

        # Load Refiner Pipeline and extract components
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        
        # --- DEFINITIVE FIX FOR REFINER DIMENSIONAL MISMATCH ---
        # The Refiner UNet's `add_embedding.linear_1` layer expects 2560 input features.
        # However, if its `config.addition_embed_type` is "text_time_ids_and_aesthetic_score",
        # the `get_aug_embed` method will try to create a 3072-dim input (1280 text + 1280 time_ids + 512 aesthetic).
        # We explicitly set `addition_embed_type` to "text_time_ids" to ensure only 2560 features are generated
        # (1280 text + 1280 time_ids, where the 7-dim time_ids tensor already includes the aesthetic score).
        print(f"Original Refiner UNet addition_embed_type: {self.unet_refiner.config.addition_embed_type}")
        if hasattr(self.unet_refiner.config, 'addition_embed_type'):
            self.unet_refiner.config.addition_embed_type = "text_time_ids"
            print(f"Set Refiner UNet addition_embed_type to: {self.unet_refiner.config.addition_embed_type}")
        
        # While related, `requires_aesthetics_score` mainly controls validation checks,
        # but the actual concatenation logic in `get_aug_embed` depends on `addition_embed_type`.
        # Setting it to False for good measure, though the `addition_embed_type` change is key.
        if hasattr(self.unet_refiner.config, 'requires_aesthetics_score'):
            self.unet_refiner.config.requires_aesthetics_score = False
        # --- END DEFINITIVE FIX ---

        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device) # Refiner only uses text_encoder_2
        self.tokenizer_refiner = pipe_refiner.tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)

        # VAE is often shared, but typically loaded in fp32 for stability unless on CPU.
        # We'll stick to the pipeline's dtype for consistency unless explicitly CPU.
        vae_load_dtype = torch.float32 if self._target_device == "cpu" else self.torch_dtype
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=vae_load_dtype).to(self._target_device)
        
        # Free up memory from temporary pipeline objects
        del pipe_base
        del pipe_refiner

        # --- Compile Models (if enabled and on CUDA) ---
        if compile_models:
            print("Compiling models with torch.compile (mode='default', fullgraph=True)... This may take a while.")
            try:
                # Compile UNets for denoising loop
                self.unet_base = torch.compile(self.unet_base, mode="default", fullgraph=True)
                self.unet_refiner = torch.compile(self.unet_refiner, mode="default", fullgraph=True)
                # Compile VAE decoder for image generation
                self.vae.decode = torch.compile(self.vae.decode, mode="default", fullgraph=True)
                # Compile Text Encoders for prompt encoding
                # These are usually smaller and might not yield as much gain, but can be included.
                self.text_encoder_base = torch.compile(self.text_encoder_base, mode="default", fullgraph=True)
                self.text_encoder_2_base = torch.compile(self.text_encoder_2_base, mode="default", fullgraph=True)
                self.text_encoder_refiner = torch.compile(self.text_encoder_refiner, mode="default", fullgraph=True)
                print("Models compiled successfully.")
            except Exception as e:
                print(f"WARNING: torch.compile failed: {e}. Running without compilation.")
                compile_models = False # Reset flag if compilation fails

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_prompt_helper(
        self,
        text_encoder: Optional[torch.nn.Module],
        tokenizer: Any,
        text_encoder_2: torch.nn.Module,
        tokenizer_2: Any,
        unet_model: torch.nn.Module,
        scheduler_model: Any,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A unified helper to encode prompts by temporarily constructing a minimal pipeline,
        ensuring correct handling of SDXL's dual text encoders, mimicking diffusers'
        encode_prompt behavior.
        """
        # Determine pipeline class based on whether text_encoder (CLIPTextModel) is used.
        # Base uses both text_encoder and text_encoder_2. Refiner uses only text_encoder_2.
        pipeline_class = StableDiffusionXLPipeline if text_encoder is not None else StableDiffusionXLImg2ImgPipeline
        
        # Temporarily create a pipeline instance (only with necessary components for encode_prompt)
        temp_pipe = pipeline_class(
            vae=self.vae, # VAE is needed to satisfy pipeline init, but not for encode_prompt
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet_model, # UNet is needed to satisfy pipeline init
            scheduler=scheduler_model # Scheduler is needed to satisfy pipeline init
        )

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt,
                device=self._target_device,
                num_images_per_prompt=1, # Assume single image per prompt for simplicity
                do_classifier_free_guidance=True, # Always prepare for CFG
                negative_prompt=negative_prompt
            )
        
        del temp_pipe # Clean up temporary object to free memory
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _get_add_time_ids(
        self, 
        original_size: Tuple[int, int], 
        crops_coords_top_left: Tuple[int, int], 
        target_size: Tuple[int, int], 
        aesthetic_score: Optional[float] = None, # Used only for Refiner (makes it 7-dim)
        negative_aesthetic_score: Optional[float] = None, # Used only for Refiner's negative aesthetic
        dtype: torch.dtype = torch.float32,
        batch_size: int = 1,
        do_classifier_free_guidance: bool = True
    ) -> torch.Tensor:
        """
        Generates `add_time_ids` for SDXL UNet conditioning, mirroring `diffusers`' logic.

        For Base UNet: Returns a 2x6 tensor (original_size, crops_coords, target_size).
        For Refiner UNet: Returns a 2x7 tensor (original_size, crops_coords, target_size, aesthetic_score).
        The first row is for negative prompt, second for positive.
        """
        add_time_ids_list = list(original_size + crops_coords_top_left + target_size)
        
        add_time_ids_pos = torch.tensor([add_time_ids_list], device=self._target_device, dtype=dtype)
        
        # If aesthetic_score is provided, it's for the Refiner, so we append it
        # and create both positive and negative aesthetic tensors.
        if aesthetic_score is not None and negative_aesthetic_score is not None:
            add_time_ids_pos = torch.cat([add_time_ids_pos, torch.tensor([[aesthetic_score]], device=self._target_device, dtype=dtype)], dim=-1)
            add_time_ids_neg = torch.cat([torch.tensor([add_time_ids_list], device=self._target_device, dtype=dtype), torch.tensor([[negative_aesthetic_score]], device=self._target_device, dtype=dtype)], dim=-1)
        else: # For Base, no aesthetic score in time_ids. Negative time_ids are same as positive.
            add_time_ids_neg = add_time_ids_pos.clone()

        if do_classifier_free_guidance:
            # Concatenate for batch processing (unconditional + conditional)
            final_add_time_ids = torch.cat([add_time_ids_neg, add_time_ids_pos], dim=0)
        else:
            final_add_time_ids = add_time_ids_pos

        # Repeat for the actual batch_size (number of images generated)
        final_add_time_ids = final_add_time_ids.repeat(batch_size, 1) if batch_size > 1 else final_add_time_ids

        return final_add_time_ids


    @torch.no_grad() # Ensure no gradients are computed for the entire __call__ method unless explicitly enabled (e.g., for Gloss)
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
        callback_on_step_end: Optional[Callable] = None, # Callback for each denoising step
        callback_on_step_end_kwargs_base: Optional[Dict[str, Any]] = None, # Extra kwargs for base callback
        callback_on_step_end_kwargs_refiner: Optional[Dict[str, Any]] = None, # Extra kwargs for refiner callback
        gloss_calculator: Optional[Any] = None, # Using Any to avoid import issues if not defined
        gloss_target: Optional[str] = None, # Target string for Gloss
        gloss_strength: float = 0.0, # Strength of Gloss guidance
        gloss_gradient_clip_norm: float = 1.0, # Gradient clipping for Gloss
        gloss_active_start_step: int = 0, # Step from which Gloss becomes active (shared for base/refiner)
        output_type: str = "pil", # "pil" for PIL Image, "np" for numpy array, "pt" for torch tensor
        aesthetic_score_refiner_pos: float = 6.0, # Positive aesthetic score for refiner's time_ids
        aesthetic_score_refiner_neg: float = 2.5, # Negative aesthetic score for refiner's time_ids
        **kwargs, # Catch any extra unused arguments to allow for future flexibility without breaking
    ) -> Dict[str, Union[List[Image.Image], Any]]:
        """
        Runs the MUE diffusion generation process.
        """
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # Set Gloss target if a calculator is provided
        if gloss_calculator:
            if gloss_target is None:
                raise ValueError("`gloss_target` must be provided if `gloss_calculator` is enabled.")
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        print("Encoding prompts for Base and Refiner models...")
        # Encode prompts for the Base pipeline
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_prompt_helper(
            self.text_encoder_base, self.tokenizer_base,
            self.text_encoder_2_base, self.tokenizer_2_base,
            self.unet_base, self.scheduler_base,
            prompt, negative_prompt
        )
        # Encode prompts for the Refiner pipeline
        # Note: text_encoder and tokenizer are None as refiner only uses text_encoder_2
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_prompt_helper(
            None, None,
            self.text_encoder_refiner, self.tokenizer_refiner,
            self.unet_refiner, self.scheduler_refiner,
            prompt, negative_prompt
        )
        print("Prompt encoding complete.")

        # Offload text encoders to CPU to free GPU memory (used only for encoding)
        if "cuda" in self._target_device:
            if self.text_encoder_base is not None:
                self.text_encoder_base.to("cpu")
            if hasattr(self, 'text_encoder_2_base'):
                self.text_encoder_2_base.to("cpu")
            if hasattr(self, 'text_encoder_refiner'):
                self.text_encoder_refiner.to("cpu")
            torch.cuda.empty_cache()
            print("Text encoders offloaded to CPU to free GPU memory.")

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet_base.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self._target_device,
            dtype=self.vae.dtype
        )
        latents *= self.scheduler_base.init_noise_sigma # Scale latents by initial noise sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        # Prepare conditioning for the Base UNet for Classifier-Free Guidance (CFG)
        prompt_embeds_b_cfg = torch.cat([neg_p_embeds_b, prompt_embeds_b], dim=0) # Concatenate uncond and cond
        add_text_embeds_b_cfg = torch.cat([neg_pooled_embeds_b, pooled_embeds_b], dim=0)

        # Get 6-dim time_ids for Base UNet (spatial info only, no aesthetic score)
        add_time_ids_base = self._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            aesthetic_score=None, # Explicitly None for Base
            negative_aesthetic_score=None, # Explicitly None for Base
            dtype=self.torch_dtype,
            batch_size=batch_size,
            do_classifier_free_guidance=True
        )

        final_added_cond_kwargs_base = {
            "text_embeds": add_text_embeds_b_cfg,
            "time_ids": add_time_ids_base
        }
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        print(f"Starting Base Denoising Loop ({base_denoise_end_step} steps)...")
        # Removed 'desc' to fix TypeError
        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            # Prepare latent input for UNet (CFG batch)
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True) # Enable gradients for Gloss calculation

            # UNet prediction for base model
            noise_pred = self.unet_base(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_b_cfg,
                added_cond_kwargs=final_added_cond_kwargs_base,
            ).sample # Access the 'sample' attribute from UNetOutput

            if is_gloss_active:
                 latents.requires_grad_(False) # Disable gradients after UNet call

            # Perform Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply Gloss if active
            if is_gloss_active:
                with torch.enable_grad(): # Re-enable gradients for the Gloss calculation
                     latents.requires_grad_(True)
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latents=latents,
                         noise_pred=guided_noise_pred, # Use the guided noise pred
                         current_timestep=t,
                         scheduler=self.scheduler_base,
                         vae_decoder=self.vae,
                         gloss_strength=gloss_strength,
                         gradient_clip_norm=gloss_gradient_clip_norm
                     )
                     latents.requires_grad_(False) # Disable gradients again after Gloss
                latents = latents.detach() - gloss_grad # Apply gradient update
            else:
                latents = latents.detach() # Ensure latents are detached if Gloss is not active

            # Scheduler step to denoise latents
            latents = self.scheduler_base.step(guided_noise_pred, t, latents, generator=generator).prev_sample

            # Callback on step end
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale # Update guidance scale if callback returns one

        print("Base Denoising Loop finished.")

        # 4. Refiner Denoising Loop
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self._target_device)
        # Determine the start timestep for the refiner based on denoising_start
        refiner_start_timestep_idx = int(len(self.scheduler_refiner.timesteps) * (1 - denoising_start))
        timesteps_refiner = self.scheduler_refiner.timesteps[refiner_start_timestep_idx:]

        # Prepare conditioning for the Refiner UNet for Classifier-Free Guidance (CFG)
        prompt_embeds_r_cfg = torch.cat([neg_p_embeds_r, prompt_embeds_r], dim=0)
        add_text_embeds_r_cfg = torch.cat([neg_pooled_embeds_r, pooled_embeds_r], dim=0)

        # Get 7-dim time_ids for Refiner UNet (spatial info + aesthetic score)
        # This is CRUCIAL for the Refiner.
        add_time_ids_refiner = self._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            aesthetic_score=aesthetic_score_refiner_pos, # Pass positive aesthetic score
            negative_aesthetic_score=aesthetic_score_refiner_neg, # Pass negative aesthetic score
            dtype=self.torch_dtype,
            batch_size=batch_size,
            do_classifier_free_guidance=True
        )

        # Final `added_cond_kwargs` for refiner: ONLY `text_embeds` and `time_ids` (7-dim)
        # Do NOT add a separate "aesthetic_score" key here. The UNet handles its internal projection
        # based on `addition_embed_type`. With `addition_embed_type="text_time_ids"`, it won't look for it.
        final_added_cond_kwargs_refiner = {
            "text_embeds": add_text_embeds_r_cfg,
            "time_ids": add_time_ids_refiner # This 2x7 tensor already includes aesthetic scores
        }
        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        print(f"Starting Refiner Denoising Loop ({len(timesteps_refiner)} steps)...")
        # Removed 'desc' to fix TypeError
        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            # Prepare latent input for UNet (CFG batch)
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # Gloss active for refiner also
            is_gloss_active = gloss_calculator and (i + base_denoise_end_step) >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True)

            # UNet prediction for refiner model
            noise_pred = self.unet_refiner(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_r_cfg,
                added_cond_kwargs=final_added_cond_kwargs_refiner,
            ).sample # Access the 'sample' attribute from UNetOutput

            if is_gloss_active:
                 latents.requires_grad_(False)

            # Perform Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply Gloss if active
            if is_gloss_active:
                with torch.enable_grad():
                    latents.requires_grad_(True)
                    gloss_grad = gloss_calculator.calculate_gloss_gradient(
                        latents=latents,
                        noise_pred=guided_noise_pred,
                        current_timestep=t,
                        scheduler=self.scheduler_refiner,
                        vae_decoder=self.vae,
                        gloss_strength=gloss_strength,
                        gradient_clip_norm=gloss_gradient_clip_norm
                    )
                    latents.requires_grad_(False)
                latents = latents.detach() - gloss_grad
            else:
                latents = latents.detach()

            # Scheduler step to denoise latents
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents, generator=generator).prev_sample
            
            # Callback on step end
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i + base_denoise_end_step, # Ensure global step count is passed
                    timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_refiner)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        print("Refiner Denoising Loop finished.")

        # 5. VAE Decoding
        print("Decoding latents to image...")
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # Post-processing using Diffusers' VaeImageProcessor
        # Initialize if not already present (e.g., if DiffusionPipeline init doesn't set it)
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            from diffusers.image_processor import VaeImageProcessor
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.scaling_factor)

        image = self.image_processor.postprocess(image, output_type=output_type)
        print("Image decoding complete.")
        
        return {"images": image}
