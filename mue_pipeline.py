import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from typing import Callable, List, Optional, Union, Dict, Any, Tuple

# Assuming these modules are correctly implemented and available
# You would need to ensure 'dis_module.py' and 'gloss_module.py' are in your project
# and contain the expected classes/functions.
from dis_module import DISCalculator
from gloss_module import GlossCalculator

class MUEDiffusionPipeline(DiffusionPipeline):
    """
    A self-contained MUE pipeline with a manual denoising loop for adaptive guidance (DIS)
    and gradient-based steering (Gloss).
    (Version 4.3 - Final, Hardened, and Architecturally Correct)

    This version incorporates robust device handling, correctly passes UNet/Scheduler
    to the internal prompt encoding helper, and crucially, accurately constructs
    the `added_cond_kwargs` for the SDXL Refiner UNet, specifically handling
    the aesthetic score's inclusion within the `time_ids` tensor.
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
                    # Handle cases like "cuda:1" when only "cuda:0" exists
                    if ":" in _requested_device:
                        device_id_str = _requested_device.split(":")[1]
                        if not device_id_str.isdigit(): # Ensure it's a valid number
                            raise ValueError(f"Invalid CUDA device ID: {device_id_str}")
                        device_id = int(device_id_str)
                        if device_id >= torch.cuda.device_count():
                            print(f"WARNING: CUDA device '{_requested_device}' specified but only {torch.cuda.device_count()} CUDA devices found. Falling back to 'cuda:0'.")
                            self._target_device = "cuda:0"
                        else:
                            self._target_device = _requested_device
                    else: # Just 'cuda' implies 'cuda:0'
                        self._target_device = "cuda:0"
                except ValueError as e:
                    print(f"WARNING: Invalid CUDA device string '{_requested_device}' ({e}). Falling back to 'cuda:0'.")
                    self._target_device = "cuda:0"
        else: # Not a CUDA device (e.g., 'cpu')
            self._target_device = _requested_device

        # Adjust dtype if running on CPU or if fp16 is not recommended/supported
        if self._target_device == "cpu":
            if _requested_torch_dtype == torch.float16:
                print("WARNING: float16 is not recommended for CPU. Changing pipeline dtype to float32.")
            self.torch_dtype = torch.float32
            if compile_models:
                print("WARNING: Disabling torch.compile as pipeline is running on CPU (minimal performance benefit/potential instability).")
                compile_models = False # Disable compilation on CPU
        else:
            self.torch_dtype = _requested_torch_dtype

        print(f"Initializing MUEDiffusionPipeline with device: {self._target_device}, dtype: {self.torch_dtype}, compile_models: {compile_models}...")

        # Common kwargs for model loading
        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True # Helps with memory during loading
        }
        # Use fp16 variant if on CUDA and using float16
        if self._target_device.startswith("cuda") and self.torch_dtype == torch.float16:
            model_loading_kwargs["variant"] = "fp16"
        else:
            # Ensure 'variant' is not passed if not applicable to avoid warnings/errors
            model_loading_kwargs.pop("variant", None)

        # --- Load Components ---
        print("Loading pipeline components...")
        # Load Stable Diffusion XL Base components
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device) # CLIPTextModel
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device) # CLIPTextModelWithProjection
        self.tokenizer_base = pipe_base.tokenizer # CLIPTokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2 # CLIPTokenizer
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        del pipe_base # Free up memory as components are moved

        # Load Stable Diffusion XL Refiner components
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        # Refiner only uses CLIPTextModelWithProjection (text_encoder_2) for its text conditioning
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device)
        self.tokenizer_refiner = pipe_refiner.tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner # Free up memory

        # Load VAE (shared between base and refiner)
        # VAE is often loaded in fp32 for stability, but we respect the pipeline's dtype unless on CPU
        vae_load_dtype = torch.float32 if self._target_device == "cpu" else self.torch_dtype
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=vae_load_dtype).to(self._target_device)

        # --- Compile Models (if enabled and on CUDA) ---
        if compile_models:
            print("Compiling models with torch.compile (mode='default', fullgraph=True)... This may take a while.")
            try:
                # Using fullgraph=True for maximum optimization, but can cause issues.
                # If errors occur, remove fullgraph=True or try mode="reduce-overhead".
                self.unet_base = torch.compile(self.unet_base, mode="default", fullgraph=True)
                self.unet_refiner = torch.compile(self.unet_refiner, mode="default", fullgraph=True)
                self.vae.decode = torch.compile(self.vae.decode, mode="default", fullgraph=True)
                # Text encoders are typically smaller and might not benefit as much, but can be compiled
                self.text_encoder_base = torch.compile(self.text_encoder_base, mode="default", fullgraph=True)
                self.text_encoder_2_base = torch.compile(self.text_encoder_2_base, mode="default", fullgraph=True)
                self.text_encoder_refiner = torch.compile(self.text_encoder_refiner, mode="default", fullgraph=True)
                print("Models compiled successfully.")
            except Exception as e:
                print(f"WARNING: torch.compile failed: {e}. Running without compilation.")
                # Fallback: do not compile if an error occurs
                compile_models = False # Reset flag if compilation failed

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_prompt_helper(
        self,
        text_encoder: Optional[torch.nn.Module], # CLIPTextModel for base, None for refiner
        tokenizer: Any, # CLIPTokenizer for base, None for refiner
        text_encoder_2: torch.nn.Module, # CLIPTextModelWithProjection (always used)
        tokenizer_2: Any, # CLIPTokenizer (always used)
        unet_model: torch.nn.Module, # The specific UNet for this stage (base or refiner)
        scheduler_model: Any, # The specific Scheduler for this stage (base or refiner)
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A unified helper to encode prompts using a temporary pipeline, ensuring correctness
        for both base and refiner stages. This is crucial as Diffusers' `encode_prompt`
        method often relies on pipeline attributes for correct behavior.
        """
        # Determine pipeline class based on whether text_encoder (CLIPTextModel) is used.
        # StableDiffusionXLPipeline uses both text encoders. StableDiffusionXLImg2ImgPipeline (refiner)
        # primarily uses text_encoder_2 for conditioning, and its `encode_prompt` simplifies for it.
        pipeline_class = StableDiffusionXLPipeline if text_encoder is not None else StableDiffusionXLImg2ImgPipeline
        
        # Initialize a temporary pipeline instance.
        # It's important to pass the correct UNet and scheduler specific to the stage
        # (base or refiner) being encoded, even if `encode_prompt` itself doesn't
        # directly use them, to maintain internal pipeline consistency.
        temp_pipe = pipeline_class(
            vae=self.vae, # VAE is shared
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet_model,       # Pass the specific UNet for this stage
            scheduler=scheduler_model # Pass the specific Scheduler for this stage
        )

        # Call the actual encode_prompt method.
        # This method handles tokenization, embedding generation, and concatenates outputs.
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt,
                device=self._target_device,
                num_images_per_prompt=1, # Fixed to 1 as current pipeline doesn't batch images
                do_classifier_free_guidance=True, # Always enable CFG
                negative_prompt=negative_prompt
            )
        
        del temp_pipe # Clean up the temporary pipeline instance to free memory
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

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
        gloss_calculator: Optional[GlossCalculator] = None, # Instance of GlossCalculator
        gloss_target: Optional[str] = None, # Target string for Gloss
        gloss_strength: float = 0.0, # Strength of Gloss guidance
        gloss_gradient_clip_norm: float = 1.0, # Gradient clipping for Gloss
        gloss_active_start_step: int = 0, # Step from which Gloss becomes active
        output_type: str = "pil", # "pil" for PIL Image, "np" for numpy array, "pt" for torch tensor
        **kwargs, # Catch any extra unused arguments to allow for future flexibility without breaking
    ) -> Dict[str, Union[List[Image.Image], Any]]:
        """
        Runs the MUE diffusion generation process.

        Args:
            prompt (Union[str, List[str]]): The prompt(s) to guide the image generation.
            negative_prompt (Optional[Union[str, List[str]]]): The negative prompt(s) to steer away from.
            height (int): The height of the generated image.
            width (int): The width of the generated image.
            num_inference_steps_base (int): Number of denoising steps for the base model.
            num_inference_steps_refiner (int): Number of denoising steps for the refiner model.
            denoising_end (float): Fraction of base denoising steps when refiner takes over.
            denoising_start (float): Fraction of refiner denoising steps to start from.
            initial_guidance_scale (float): Initial classifier-free guidance scale.
            seed (int): Random seed for reproducibility.
            callback_on_step_end (Optional[Callable]): A function to call at the end of each denoising step.
                It can modify `current_guidance_scale`.
            callback_on_step_end_kwargs_base (Optional[Dict[str, Any]]): Additional keyword arguments
                to pass to the callback during base denoising.
            callback_on_step_end_kwargs_refiner (Optional[Dict[str, Any]]): Additional keyword arguments
                to pass to the callback during refiner denoising.
            gloss_calculator (Optional[GlossCalculator]): An instance of `GlossCalculator` for gradient steering.
            gloss_target (Optional[str]): The target concept for Gloss guidance.
            gloss_strength (float): The strength of the Gloss guidance.
            gloss_gradient_clip_norm (float): Gradient clipping norm for Gloss.
            gloss_active_start_step (int): The step at which Gloss guidance becomes active.
            output_type (str): The desired output format ("pil", "np", "pt").
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Union[List[Image.Image], Any]]: A dictionary containing the generated images.
        """
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # Set Gloss target if a calculator is provided
        if gloss_calculator:
            if gloss_target is None:
                raise ValueError("gloss_target must be provided if gloss_calculator is enabled.")
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        # Encode prompts for the Base pipeline
        # `text_encoder_base` and `tokenizer_base` are for CLIPTextModel (text_encoder_1)
        # `text_encoder_2_base` and `tokenizer_2_base` are for CLIPTextModelWithProjection (text_encoder_2)
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_prompt_helper(
            self.text_encoder_base, self.tokenizer_base,
            self.text_encoder_2_base, self.tokenizer_2_base,
            self.unet_base, self.scheduler_base, # Pass base's UNet and Scheduler
            prompt, negative_prompt
        )
        
        # Encode prompts for the Refiner pipeline
        # The refiner's encode_prompt uses only `text_encoder_2` (CLIPTextModelWithProjection)
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_prompt_helper(
            None, None, # text_encoder_1 and its tokenizer are not used by the refiner's prompt encoding
            self.text_encoder_refiner, self.tokenizer_refiner,
            self.unet_refiner, self.scheduler_refiner, # Pass refiner's UNet and Scheduler
            prompt, negative_prompt
        )

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet_base.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self._target_device,
            dtype=self.vae.dtype # Use VAE's dtype for latents
        )
        latents *= self.scheduler_base.init_noise_sigma # Scale latents by initial noise sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        # Prepare conditioning for the Base UNet for Classifier-Free Guidance (CFG)
        # `encoder_hidden_states` is the concatenated output of both text encoders
        prompt_embeds_b_cfg = torch.cat([neg_p_embeds_b, prompt_embeds_b], dim=0)

        # `added_cond_kwargs` for the base UNet typically includes `text_embeds` (pooled output from text_encoder_2)
        # and `time_ids` (original_size, crops_coords, target_size).
        add_text_embeds_b_cfg = torch.cat([neg_pooled_embeds_b, pooled_embeds_b], dim=0)

        # SDXL's time_ids are a 6-dim tensor: [original_height, original_width, crop_top, crop_left, target_height, target_width]
        original_size = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype)
        crops_coords_top_left = torch.tensor([[0, 0]], device=self._target_device, dtype=self.torch_dtype)
        target_size = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype)
        
        # Combine for a single batch entry, then repeat for batch_size
        add_time_ids_b = torch.cat([original_size, crops_coords_top_left, target_size], dim=-1).repeat(batch_size, 1)
        add_time_ids_b_cfg = torch.cat([add_time_ids_b, add_time_ids_b], dim=0) # Duplicate for CFG

        # Final dictionary for UNet's `added_cond_kwargs`
        final_added_cond_kwargs_base = {
            "text_embeds": add_text_embeds_b_cfg, # Shape: (2*batch_size, 1280)
            "time_ids": add_time_ids_b_cfg        # Shape: (2*batch_size, 6)
        }
        # Prepare keyword arguments for the callback function
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        print(f"Starting Base Denoising Loop ({base_denoise_end_step} steps)...")
        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step], desc="Base Denoising")):
            # Prepare latent input for UNet (duplicated for CFG)
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            # Check if Gloss guidance should be active for this step
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                # Enable gradient tracking on latents for Gloss calculation
                latents.requires_grad_(True)

            # --- UNET BASE CALL ---
            # Predict noise from latents
            noise_pred = self.unet_base(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_b_cfg,
                added_cond_kwargs=final_added_cond_kwargs_base
            ).sample
            
            # Disable gradient tracking on latents immediately after UNet call
            if is_gloss_active:
                 latents.requires_grad_(False)

            # Apply Classifier-Free Guidance (CFG)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply Gloss (if active)
            if is_gloss_active:
                with torch.enable_grad(): # Re-enable grad specifically for gloss calculation, in a controlled context
                     latents.requires_grad_(True)
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latents, guided_noise_pred, t, self.scheduler_base, self.vae, gloss_strength, gloss_gradient_clip_norm)
                     latents.requires_grad_(False) # Disable grad again
                # Update latents with the gloss correction
                latents = latents.detach() - gloss_grad

            # Scheduler step: denoise latents using the predicted noise
            latents = self.scheduler_base.step(guided_noise_pred, t, latents).prev_sample

            # Execute callback function if provided
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale # Update guidance scale if returned by callback

        # 4. Refiner Denoising Loop
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self._target_device)
        # Calculate the starting timestep for the refiner based on denoising_start
        refiner_start_idx = int(num_inference_steps_refiner * (1 - denoising_start))
        timesteps_refiner = self.scheduler_refiner.timesteps[refiner_start_idx:]
        
        # Prepare conditioning for the Refiner UNet for CFG
        # Refiner's `encoder_hidden_states` comes from `text_encoder_refiner` (CLIPTextModelWithProjection)
        # Its shape is (2*batch_size, sequence_length, 1280)
        prompt_embeds_r_cfg = torch.cat([neg_p_embeds_r, prompt_embeds_r], dim=0)

        # --- REFINER'S `added_cond_kwargs` (CRITICAL FIX) ---
        # Refiner's `added_cond_kwargs` requires `text_embeds` (pooled output from text_encoder_2)
        # and `time_ids`. Crucially, for the refiner, the `aesthetic_score` is CONCATENATED
        # directly into the `time_ids` tensor, making it 7-dimensional.
        
        # 1. `text_embeds`: This is the pooled output from text_encoder_2
        add_text_embeds_r_cfg = torch.cat([neg_pooled_embeds_r, pooled_embeds_r], dim=0) # Shape: (2*batch_size, 1280)

        # 2. `time_ids`: This must be a 7-dimensional tensor.
        # It consists of [original_height, original_width, crop_top, crop_left, target_height, target_width, aesthetic_score]
        original_size_r = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype)
        crops_coords_top_left_r = torch.tensor([[0, 0]], device=self._target_device, dtype=self.torch_dtype)
        target_size_r = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype)
        
        # Default aesthetic scores for refiner (can be made configurable if needed)
        # Positive aesthetic score for positive prompt, negative for negative prompt
        aesthetic_score_pos = torch.tensor([[6.0]], device=self._target_device, dtype=self.torch_dtype)
        aesthetic_score_neg = torch.tensor([[2.5]], device=self._target_device, dtype=self.torch_dtype) # Lower aesthetic score for negative

        # Concatenate spatial info and aesthetic score for `time_ids`
        # Create a single `time_ids` entry for positive and negative prompts
        add_time_ids_r_pos = torch.cat([original_size_r, crops_coords_top_left_r, target_size_r, aesthetic_score_pos], dim=-1).repeat(batch_size, 1) # Shape: (batch_size, 7)
        add_time_ids_r_neg = torch.cat([original_size_r, crops_coords_top_left_r, target_size_r, aesthetic_score_neg], dim=-1).repeat(batch_size, 1) # Shape: (batch_size, 7)
        
        # Combine for CFG (negative first, then positive)
        add_time_ids_r_cfg = torch.cat([add_time_ids_r_neg, add_time_ids_r_pos], dim=0) # Shape: (2*batch_size, 7)

        # The refiner's `added_cond_kwargs` structure (as per diffusers `StableDiffusionXLImg2ImgPipeline` source):
        final_added_cond_kwargs_refiner = {
            "text_embeds": add_text_embeds_r_cfg, # Shape: (2*batch_size, 1280)
            "time_ids": add_time_ids_r_cfg        # Shape: (2*batch_size, 7)
        }
        # Note: A separate "aesthetic_score" key is NOT used here; it's bundled into "time_ids".
        # --- END REFINER'S `added_cond_kwargs` FIX ---

        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        print(f"Starting Refiner Denoising Loop ({len(timesteps_refiner)} steps)...")
        for i, t in enumerate(self.progress_bar(timesteps_refiner, desc="Refiner Denoising")):
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # --- UNET REFINER CALL ---
            noise_pred = self.unet_refiner(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_r_cfg,
                added_cond_kwargs=final_added_cond_kwargs_refiner
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents).prev_sample
            
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i + base_denoise_end_step, # Adjust step count for refiner to be continuous with base
                    timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_refiner)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        # 5. VAE Decoding
        # Scale latents back to image space before decoding
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        # Post-process the raw image tensor into the desired output format (PIL, NumPy, or Torch tensor)
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return {"images": image}