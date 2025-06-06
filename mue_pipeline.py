# File: mue_pipeline.py

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
    and gradient-based steering (Gloss). Optimized with torch.compile.

    This version is hardened against common environmental and architectural flaws,
    featuring dynamic device/dtype handling and accurate conditioning for SDXL models.
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        """
        Initializes the MUEDiffusionPipeline.

        Args:
            device (str): The target device for model execution (e.g., "cuda", "cuda:0", "cpu").
                          Automatically falls back to CPU if CUDA is requested but unavailable.
            torch_dtype (torch.dtype): The desired torch data type for model weights and computations.
                                       Automatically set to torch.float32 if running on CPU.
            compile_models (bool): If True, attempts to compile models using torch.compile for performance.
                                   Automatically disabled if running on CPU.
        """
        super().__init__()

        # --- Dynamic Device and Dtype Handling ---
        # Store requested device and dtype for initial evaluation
        _requested_device = device
        _requested_torch_dtype = torch_dtype

        # Determine the actual device to use
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
                except ValueError: # Handles cases like "cuda:abc" if provided
                    print(f"WARNING: Invalid CUDA device string '{_requested_device}'. Falling back to 'cuda:0'.")
                    self._target_device = "cuda:0"
            else: # "cuda" implies "cuda:0"
                self._target_device = "cuda"
        else: # Explicitly "cpu" or other non-CUDA string
            self._target_device = _requested_device

        # Adjust dtype based on the final determined device
        if self._target_device == "cpu":
            if _requested_torch_dtype == torch.float16:
                print("WARNING: float16 is not recommended for CPU. Changing pipeline dtype to float32.")
            self.torch_dtype = torch.float32 # Force float32 for CPU for compatibility and performance
            if compile_models:
                print("WARNING: Disabling torch.compile as pipeline is running on CPU (minimal benefit/potential instability).")
                compile_models = False # torch.compile is typically for GPU performance
        else: # GPU execution
            self.torch_dtype = _requested_torch_dtype # Use the requested dtype for GPU

        # --- Model Loading Keyword Arguments ---
        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        # Only add 'fp16' variant if running on CUDA with float16 to load optimized weights
        if self._target_device.startswith("cuda") and self.torch_dtype == torch.float16:
            model_loading_kwargs["variant"] = "fp16"
        else:
            model_loading_kwargs.pop("variant", None) # Remove variant if not applicable

        print(f"Initializing MUEDiffusionPipeline with device: {self._target_device}, dtype: {self.torch_dtype}, compile_models: {compile_models}...")

        # --- Load Base Pipeline Components ---
        # Load Stable Diffusion XL Base model components
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        del pipe_base # Release memory from the full pipeline object

        # --- Load Refiner Pipeline Components ---
        # Using StableDiffusionXLImg2ImgPipeline as it's commonly used for the refiner
        # and its encode_prompt behavior aligns with refiner needs.
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device) # Refiner primarily uses text_encoder_2
        self.tokenizer_refiner = pipe_refiner.tokenizer_2 # Corresponding tokenizer
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner # Release memory

        # --- Load VAE (Variational AutoEncoder) ---
        # The 'fp16-fix' VAE is optimized for FP16. Load as FP32 if on CPU.
        vae_load_dtype = torch.float32 if self._target_device == "cpu" else self.torch_dtype
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=vae_load_dtype).to(self._target_device)

        # --- Compile Models if enabled ---
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
        """
        Encodes prompts for the Stable Diffusion XL Base model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        """
        # Temporarily instantiate pipeline to use its `encode_prompt` method
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
        del temp_pipe # Clean up temporary pipeline
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_refiner_prompt(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes prompts for the Stable Diffusion XL Refiner model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                prompt_embeds (3D), negative_prompt_embeds (3D),
                pooled_prompt_embeds (2D), negative_pooled_prompt_embeds (2D)
        """
        # Temporarily instantiate pipeline to use its `encode_prompt` method.
        # Note: Refiner only uses text_encoder_2 and its associated tokenizer.
        temp_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.vae, text_encoder=None, tokenizer=None,
            text_encoder_2=self.text_encoder_refiner, tokenizer_2=self.tokenizer_refiner,
            unet=self.unet_refiner, scheduler=self.scheduler_refiner
        )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt, device=self._target_device, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )
        del temp_pipe # Clean up temporary pipeline
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _prepare_added_cond_kwargs_base(self, height: int, width: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Prepares additional conditioning arguments for the Base UNet.
        Includes image dimensions as 'time_ids'.
        """
        # SDXL's UNet expects 'time_ids' for resolution conditioning
        time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype)
        return {"time_ids": time_ids.repeat(batch_size, 1)}

    def _prepare_added_cond_kwargs_refiner(self, height: int, width: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Prepares additional conditioning arguments for the Refiner UNet.
        Includes image dimensions ('time_ids') and aesthetic embeddings.
        """
        # Refiner UNet also expects 'time_ids' and typically uses aesthetic embeddings
        time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=self._target_device, dtype=self.torch_dtype)
        add_aesthetic_embeds = torch.tensor([[2.5, 2.5, 2.5, 2.5, 2.5, 2.5]], device=self._target_device, dtype=self.torch_dtype) # Common default
        return {"time_ids": time_ids.repeat(batch_size, 1), "add_aesthetic_embeds": add_aesthetic_embeds.repeat(batch_size, 1)}

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
        """
        Generates an image using the MUE Diffusion Pipeline.

        Args:
            prompt (Union[str, List[str]]): The main prompt(s) for image generation.
            negative_prompt (Optional[Union[str, List[str]]]): The negative prompt(s).
            height (int): The height of the generated image.
            width (int): The width of the generated image.
            num_inference_steps_base (int): Number of denoising steps for the base model.
            num_inference_steps_refiner (int): Number of denoising steps for the refiner model.
            denoising_end (float): The fraction of base denoising steps at which to stop.
            denoising_start (float): The fraction of refiner denoising steps at which to start.
            initial_guidance_scale (float): Initial Classifier-Free Guidance (CFG) scale.
            seed (int): Random seed for reproducibility.
            callback_on_step_end (Optional[Callable]): A function to call at the end of each denoising step.
            callback_on_step_end_kwargs_base (Optional[Dict]): Additional kwargs for base callback.
            callback_on_step_end_kwargs_refiner (Optional[Dict]): Additional kwargs for refiner callback.
            gloss_calculator (Optional[GlossCalculator]): Instance for Gloss (gradient-based steering).
            gloss_target (Optional[str]): Target for Gloss steering.
            gloss_strength (float): Strength of Gloss steering.
            gloss_gradient_clip_norm (float): Gradient clipping norm for Gloss.
            gloss_active_start_step (int): Step from which Gloss becomes active.
            output_type (str): Desired output format ("pil" for PIL Image, "np" for numpy array, "pt" for torch tensor).
            **kwargs: Additional keyword arguments (e.g., for compatibility with other Diffusers features).

        Returns:
            Dict[str, Union[List[Image.Image], Any]]: A dictionary containing the generated images.
        """
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if gloss_calculator:
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        # Encode prompts for both base and refiner models.
        # Base uses text_encoder_1 and text_encoder_2.
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = \
            self._encode_base_prompt(prompt, negative_prompt)
        # Refiner uses text_encoder_2 only.
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = \
            self._encode_refiner_prompt(prompt, negative_prompt)

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            generator=generator,
            device=self._target_device,
            dtype=self.vae.dtype
        )
        latents *= self.scheduler_base.init_noise_sigma # Scale latents with initial noise sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        # Prepare full conditioning tensors for Classifier-Free Guidance (CFG)
        # Concatenate negative and positive embeddings for base UNet
        full_prompt_embeds_b = torch.cat([neg_p_embeds_b, prompt_embeds_b]) # (2*batch, seq_len, 2304)
        full_added_text_embeds_b = torch.cat([neg_pooled_embeds_b, pooled_embeds_b]) # (2*batch, 1280)

        # Prepare additional conditioning kwargs for base UNet
        added_cond_kwargs_b = self._prepare_added_cond_kwargs_base(height, width, batch_size) # Contains 'time_ids'
        
        # Combine all added conditioning into a single dictionary, ensuring 'time_ids' is present
        # Use dictionary unpacking to correctly merge 'text_embeds' and 'time_ids'
        final_added_cond_kwargs_base = {**added_cond_kwargs_b, "text_embeds": full_added_text_embeds_b}

        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            latent_model_input = torch.cat([latents] * 2) # Duplicate latents for CFG
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True) # Enable gradient tracking for Gloss

            # Predict noise using the base UNet
            noise_pred = self.unet_base(
                latent_model_input, t,
                encoder_hidden_states=full_prompt_embeds_b, # Long text embeddings
                added_cond_kwargs=final_added_cond_kwargs_base # Contains time_ids and pooled text embeds
            ).sample
            
            if is_gloss_active:
                 latents.requires_grad_(False) # Disable gradient tracking after UNet pass

            # Apply Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply Gloss (gradient-based steering) if active
            if is_gloss_active:
                with torch.enable_grad():
                     latents.requires_grad_(True)
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latents, guided_noise_pred, t, self.scheduler_base, self.vae, gloss_strength, gloss_gradient_clip_norm)
                     latents.requires_grad_(False)
                latents = latents.detach() - gloss_grad # Apply gradient update

            # Take a denoising step
            latents = self.scheduler_base.step(guided_noise_pred, t, latents).prev_sample

            # Execute step-end callback
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale # Update guidance scale dynamically

        # 4. Refiner Denoising Loop
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self._target_device)
        # Determine refiner timesteps based on denoising_start
        timesteps_refiner = self.scheduler_refiner.timesteps[int(num_inference_steps_refiner * (1-denoising_start)):]
        
        # --- Prepare Refiner Text Conditionings (Crucial for SDXL Refiner) ---
        # The refiner UNet typically expects two main text conditioning inputs:
        # 1. `encoder_hidden_states`: The 3D tensor from text_encoder_2's last_hidden_state (batch, seq_len, 1280).
        # 2. `added_cond_kwargs['text_embeds']`: A 2D tensor (batch, 2560) which is a concatenation of
        #    a pooled version of the 3D text_encoder_2 output AND the text_encoder_2's pooled_output.

        # 1. Prepare `encoder_hidden_states` (3D tensor for UNet's main text input)
        full_encoder_hidden_states_r = torch.cat([neg_p_embeds_r, prompt_embeds_r]) # (2*batch, seq_len, 1280)
        
        # 2. Prepare `added_cond_kwargs['text_embeds']` (2D tensor for UNet's 'added' text input)
        #   a. Pool the 3D `prompt_embeds_r` (last_hidden_state from text_encoder_2) down to 2D.
        #      Commonly, this is done by taking the embedding of the last token (EOS token).
        refiner_pooled_text_embeds = prompt_embeds_r[:, -1, :] # (batch, 1280)
        refiner_pooled_text_embeds_neg = neg_p_embeds_r[:, -1, :] # (batch, 1280) (negative)

        #   b. Concatenate this pooled 3D output with the `pooled_embeds_r` (pooled_output from text_encoder_2).
        #      This creates the (batch, 1280 + 1280) = (batch, 2560) tensor.
        full_added_text_embeds_r_combined = torch.cat([refiner_pooled_text_embeds, pooled_embeds_r], dim=-1)
        full_neg_added_text_embeds_r_combined = torch.cat([refiner_pooled_text_embeds_neg, neg_pooled_embeds_r], dim=-1)

        #   c. Concatenate positive and negative for CFG
        final_added_cond_text_embeds_r = torch.cat([full_neg_added_text_embeds_r_combined, full_added_text_embeds_r_combined])
        
        # Prepare other additional conditioning kwargs for refiner UNet (time_ids, aesthetic embeds)
        added_cond_kwargs_r = self._prepare_added_cond_kwargs_refiner(height, width, batch_size)
        
        # Combine all added conditioning into a single dictionary
        # Ensure 'time_ids' and 'add_aesthetic_embeds' are correctly merged with 'text_embeds'
        final_added_cond_kwargs_r = {**added_cond_kwargs_r, "text_embeds": final_added_cond_text_embeds_r}

        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # Predict noise using the refiner UNet
            noise_pred = self.unet_refiner(
                latent_model_input, t,
                encoder_hidden_states=full_encoder_hidden_states_r, # Long text embeddings (3D)
                added_cond_kwargs=final_added_cond_kwargs_r # Contains combined pooled text embeds (2D), time_ids, aesthetic
            ).sample
            
            # Apply Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Take a denoising step
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents).prev_sample
            
            # Execute step-end callback
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i + base_denoise_end_step, # Adjust step count for overall pipeline
                    timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_refiner)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        # 5. VAE Decoding to Image
        # Decode the final latents back into a pixel space image
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        # Post-process the image (e.g., convert to PIL format, scale)
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return {"images": image}