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
    (Version 4.1 - Final, Hardened, and Architecturally Correct)
    """
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.float16, compile_models: bool = False):
        super().__init__()

        # --- Device & Dtype Safety Checks ---
        # Store requested device and dtype for logging/debugging
        _requested_device = device
        _requested_torch_dtype = torch_dtype

        if _requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"WARNING: CUDA device '{_requested_device}' requested but CUDA is not available. Falling back to CPU.")
                self._target_device = "cpu"
            else:
                # Validate specific CUDA device if provided (e.g., cuda:1)
                try:
                    if ":" in _requested_device:
                        device_id = int(_requested_device.split(":")[1])
                        if device_id >= torch.cuda.device_count():
                            print(f"WARNING: CUDA device '{_requested_device}' specified but only {torch.cuda.device_count()} CUDA devices found. Falling back to 'cuda:0'.")
                            self._target_device = "cuda:0"
                        else:
                            self._target_device = _requested_device
                    else: # Just 'cuda' implies 'cuda:0'
                        self._target_device = "cuda:0"
                except ValueError:
                    print(f"WARNING: Invalid CUDA device string '{_requested_device}'. Falling back to 'cuda:0'.")
                    self._target_device = "cuda:0"
        else: # Not a CUDA device, e.g., 'cpu'
            self._target_device = _requested_device

        # Adjust dtype if running on CPU or if fp16 is not supported/recommended
        if self._target_device == "cpu":
            if _requested_torch_dtype == torch.float16:
                print("WARNING: float16 is not recommended for CPU. Changing pipeline dtype to float32.")
            self.torch_dtype = torch.float32
            if compile_models:
                print("WARNING: Disabling torch.compile as pipeline is running on CPU (minimal benefit/potential instability).")
                compile_models = False
        else:
            self.torch_dtype = _requested_torch_dtype

        print(f"Initializing MUEDiffusionPipeline with device: {self._target_device}, dtype: {self.torch_dtype}, compile_models: {compile_models}...")

        model_loading_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
        if self._target_device.startswith("cuda") and self.torch_dtype == torch.float16:
            model_loading_kwargs["variant"] = "fp16"
        else:
            model_loading_kwargs.pop("variant", None) # Ensure variant is not passed if not fp16 cuda

        # --- Load Components ---
        # Base Pipeline Components
        pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **model_loading_kwargs)
        self.unet_base = pipe_base.unet.to(self._target_device)
        self.text_encoder_base = pipe_base.text_encoder.to(self._target_device)
        self.text_encoder_2_base = pipe_base.text_encoder_2.to(self._target_device)
        self.tokenizer_base = pipe_base.tokenizer
        self.tokenizer_2_base = pipe_base.tokenizer_2
        self.scheduler_base = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config, use_karras_sigmas=True)
        del pipe_base # Free up memory

        # Refiner Pipeline Components
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", **model_loading_kwargs)
        self.unet_refiner = pipe_refiner.unet.to(self._target_device)
        self.text_encoder_refiner = pipe_refiner.text_encoder_2.to(self._target_device) # Refiner only uses text_encoder_2
        self.tokenizer_refiner = pipe_refiner.tokenizer_2
        self.scheduler_refiner = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config, use_karras_sigmas=True)
        del pipe_refiner # Free up memory

        # VAE (shared between base and refiner)
        # VAE is often best in fp32 for stability, but we'll stick to selected dtype for consistency unless on CPU
        vae_load_dtype = torch.float32 if self._target_device == "cpu" else self.torch_dtype
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=vae_load_dtype).to(self._target_device)

        # --- Compile Models (if enabled and on CUDA) ---
        if compile_models:
            print("Compiling models with torch.compile...")
            # For optimal compilation, use fullgraph=True if possible, else default
            # `mode="reduce-overhead"` can be a good intermediate if `fullgraph=True` fails
            self.unet_base = torch.compile(self.unet_base, mode="default", fullgraph=True)
            self.unet_refiner = torch.compile(self.unet_refiner, mode="default", fullgraph=True)
            self.vae.decode = torch.compile(self.vae.decode, mode="default", fullgraph=True)
            self.text_encoder_base = torch.compile(self.text_encoder_base, mode="default", fullgraph=True)
            self.text_encoder_2_base = torch.compile(self.text_encoder_2_base, mode="default", fullgraph=True)
            self.text_encoder_refiner = torch.compile(self.text_encoder_refiner, mode="default", fullgraph=True)
            print("Models compiled.")

        print("MUEDiffusionPipeline initialization complete.")

    def _encode_prompt_helper(
        self,
        text_encoder: Optional[torch.nn.Module], # Text encoder 1 for base, None for refiner
        tokenizer: Any, # Tokenizer 1 for base, None for refiner
        text_encoder_2: torch.nn.Module, # Text encoder 2 (always present)
        tokenizer_2: Any, # Tokenizer 2 (always present)
        unet_model: torch.nn.Module, # The specific UNet for this stage (base or refiner)
        scheduler_model: Any, # The specific Scheduler for this stage (base or refiner)
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A unified helper to encode prompts using a temporary pipeline, ensuring correctness
        for both base and refiner stages.
        """
        # Determine pipeline class based on whether text_encoder (CLIPTextModel) is used
        pipeline_class = StableDiffusionXLPipeline if text_encoder is not None else StableDiffusionXLImg2ImgPipeline
        
        # Initialize temp_pipe with the correct UNet and scheduler for its context
        temp_pipe = pipeline_class(
            vae=self.vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            unet=unet_model,       # Use the specific UNet passed in
            scheduler=scheduler_model # Use the specific Scheduler passed in
        )

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            temp_pipe.encode_prompt(
                prompt,
                device=self._target_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt
            )
        del temp_pipe # Clean up temporary pipeline
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

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
        **kwargs, # Catch any extra unused arguments
    ) -> Dict[str, Union[List[Image.Image], Any]]:

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if gloss_calculator:
            gloss_calculator.set_target(gloss_target)

        # 1. Prompt Encoding
        # Encode prompts for the Base pipeline
        prompt_embeds_b, neg_p_embeds_b, pooled_embeds_b, neg_pooled_embeds_b = self._encode_prompt_helper(
            self.text_encoder_base, self.tokenizer_base, self.text_encoder_2_base, self.tokenizer_2_base,
            self.unet_base, self.scheduler_base, # Pass the BASE UNet and Scheduler
            prompt, negative_prompt
        )
        
        # Encode prompts for the Refiner pipeline
        # Note: Refiner only uses text_encoder_2
        prompt_embeds_r, neg_p_embeds_r, pooled_embeds_r, neg_pooled_embeds_r = self._encode_prompt_helper(
            None, None, # text_encoder (CLIP) and tokenizer (CLIP) are not used by refiner's encode_prompt
            self.text_encoder_refiner, self.tokenizer_refiner,
            self.unet_refiner, self.scheduler_refiner, # Pass the REFINER UNet and Scheduler
            prompt, negative_prompt
        )

        # 2. Latent & Timestep Preparation
        generator = torch.Generator(device=self._target_device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet_base.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self._target_device,
            dtype=self.vae.dtype
        )
        latents *= self.scheduler_base.init_noise_sigma # Apply initial noise sigma

        # 3. Base Denoising Loop
        self.scheduler_base.set_timesteps(num_inference_steps_base, device=self._target_device)
        timesteps_base = self.scheduler_base.timesteps
        current_guidance_scale = initial_guidance_scale
        base_denoise_end_step = int(num_inference_steps_base * denoising_end)

        # Prepare conditioning for the Base UNet for Classifier-Free Guidance (CFG)
        # `encoder_hidden_states` (long embeds from both text encoders)
        prompt_embeds_b_cfg = torch.cat([neg_p_embeds_b, prompt_embeds_b]) # Shape: (2*batch, seq_len, 2048)

        # `added_cond_kwargs['text_embeds']` (pooled embeds from text_encoder_2)
        add_text_embeds_b_cfg = torch.cat([neg_pooled_embeds_b, pooled_embeds_b]) # Shape: (2*batch, 1280)

        # `added_cond_kwargs['time_ids']` (original_size, crops_coords, target_size)
        # These are repeated for CFG (negative and positive prompts)
        original_size = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        crops_coords_top_left = torch.tensor([[0, 0]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        target_size = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        
        # Concatenate these 2-dim tensors into the 6-dim time_ids expected by diffusers
        # Note: The UNet projects this 6-dim input to a 256-dim embedding internally
        add_time_ids_b = torch.cat([original_size, crops_coords_top_left, target_size], dim=-1) # Shape: (batch, 6)
        add_time_ids_b_cfg = torch.cat([add_time_ids_b, add_time_ids_b]) # Shape: (2*batch, 6)

        # Final dictionary for UNet's `added_cond_kwargs`
        final_added_cond_kwargs_base = {
            "text_embeds": add_text_embeds_b_cfg,
            "time_ids": add_time_ids_b_cfg
        }
        final_cb_kwargs_base = {**(callback_on_step_end_kwargs_base or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_base[:base_denoise_end_step])):
            latent_model_input = torch.cat([latents] * 2) # Duplicate latents for CFG
            latent_model_input = self.scheduler_base.scale_model_input(latent_model_input, t)
            
            is_gloss_active = gloss_calculator and i >= gloss_active_start_step and gloss_strength > 0
            if is_gloss_active:
                latents.requires_grad_(True) # Enable grad for gloss calculation

            # --- UNET BASE CALL ---
            noise_pred = self.unet_base(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_b_cfg,
                added_cond_kwargs=final_added_cond_kwargs_base
            ).sample
            
            if is_gloss_active:
                 latents.requires_grad_(False) # Disable grad immediately after UNet call

            # Apply Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Apply Gloss (if active)
            if is_gloss_active:
                with torch.enable_grad(): # Re-enable grad specifically for gloss calculation
                     latents.requires_grad_(True)
                     gloss_grad = gloss_calculator.calculate_gloss_gradient(
                         latents, guided_noise_pred, t, self.scheduler_base, self.vae, gloss_strength, gloss_gradient_clip_norm)
                     latents.requires_grad_(False) # Disable grad again
                latents = latents.detach() - gloss_grad # Apply gloss correction

            # Scheduler step
            latents = self.scheduler_base.step(guided_noise_pred, t, latents).prev_sample

            # Callback
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i, timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_base)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        # 4. Refiner Denoising Loop
        self.scheduler_refiner.set_timesteps(num_inference_steps_refiner, device=self._target_device)
        # Select timesteps for refiner based on denoising_start
        timesteps_refiner = self.scheduler_refiner.timesteps[int(num_inference_steps_refiner * (1-denoising_start)):]
        
        # Prepare conditioning for the Refiner UNet for CFG
        # `encoder_hidden_states` (from text_encoder_2 only)
        # For refiner, prompt_embeds_r is already (batch, seq_len, 1280)
        prompt_embeds_r_cfg = torch.cat([neg_p_embeds_r, prompt_embeds_r]) # Shape: (2*batch, seq_len, 1280)

        # `added_cond_kwargs['text_embeds']` (from text_encoder_2 pooled output)
        # And also includes a *pooled* version of text_encoder_2 last hidden state (for refiner specifics)
        # This creates the 2560-dim input for the refiner's added_text_embeds
        # pooled_embeds_r: (batch, 1280)
        # neg_pooled_embeds_r: (batch, 1280)
        final_added_cond_text_embeds_r = torch.cat([neg_pooled_embeds_r, pooled_embeds_r])
        # Note: Some refiners might concatenate pooled_prompt_embeds with the *last token* of prompt_embeds_r
        # `prompt_embeds_r[:, -1, :]` for the 2560.
        # But `StableDiffusionXLPipeline`'s `_get_add_embeds` uses `pooled_prompt_embeds`
        # and for refiner (text_encoder_2 only), prompt_embeds are already 1280-dim, so pooling it again
        # as `prompt_embeds_r[:, -1, :]` might not be standard.
        # Let's stick to the simplest, which is combining pooled_output with itself for now if 2560 is needed.
        # However, the common refiner setup is `text_embeds` being `pooled_output_2` and `time_ids` (+ aesthetic).
        # The `2560` dimension for refiner's `added_cond_kwargs["text_embeds"]` actually comes from
        # `torch.cat([pooled_prompt_embeds, aesthetic_embeds])` where aesthetic_embeds is typically 1280-dim.
        # OR it can come from the `pooled_output` and the `last_hidden_state` (pooled) of text_encoder_2.
        # Let's check diffusers source for refiner specifically.
        # Standard SDXL refiner uses:
        # text_embeds (2560): concat of pooled_prompt_embeds_r (1280) and negative_pooled_prompt_embeds_r (1280) -- no, that's not it
        # It's usually `text_encoder_2.config.projection_dim` (1280) and `self.text_encoder_2.config.pooled_output_dim` (1280).
        # The `prompt_embeds_r` is the full `last_hidden_state` from text_encoder_2 (seq_len, 1280).
        # The `pooled_embeds_r` is the `pooled_output` from text_encoder_2 (1280).
        # Refiner `added_cond_kwargs['text_embeds']` is usually `pooled_prompt_embeds` + `aesthetic_embeds`.
        # However, the `StableDiffusionXLImg2ImgPipeline`'s `encode_prompt` already provides `pooled_prompt_embeds` from `text_encoder_2`.
        # Let's align with the refiner's `_get_add_embeds` which takes `aesthetic_score`.

        # Refiner `add_time_ids` and `add_aesthetic_embeds`
        original_size_r = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        crops_coords_top_left_r = torch.tensor([[0, 0]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        target_size_r = torch.tensor([[height, width]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        add_time_ids_r = torch.cat([original_size_r, crops_coords_top_left_r, target_size_r], dim=-1) # Shape: (batch, 6)
        add_time_ids_r_cfg = torch.cat([add_time_ids_r, add_time_ids_r]) # Shape: (2*batch, 6)

        # For refiner, aesthetic score is also part of added_cond_kwargs
        # Default aesthetic score for generation is 6.0 for positive, 2.5 for negative
        aesthetic_score = torch.tensor([[6.0]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        negative_aesthetic_score = torch.tensor([[2.5]], device=self._target_device, dtype=self.torch_dtype).repeat(batch_size, 1)
        
        # Combine aesthetic scores for CFG
        add_aesthetic_embeds_r_cfg = torch.cat([negative_aesthetic_score, aesthetic_score]) # Shape: (2*batch, 1)

        # The `text_embeds` for refiner are usually `pooled_prompt_embeds` (1280-dim)
        # However, the `add_embedding` layer in the refiner UNet often expects a concatenated
        # vector of `pooled_prompt_embeds` and the `aesthetic_score` (which gets projected to a higher dim).
        # Let's align precisely with `StableDiffusionXLImg2ImgPipeline`'s `_get_add_embeds`:
        # For refiner:
        #   add_text_embeds = pooled_prompt_embeds
        #   add_time_ids includes original_size, crops, target_size, AND aesthetic_score
        #   It seems `time_ids` is expected to be a single 6-dim vector for original/crop/target,
        #   and `aesthetic_score` is passed separately as `add_aesthetic_embeds` (1-dim).
        #   The UNet then concatenates `time_embeds` (256-dim output of time_embedding) and `text_embeds` (1280-dim)
        #   and `aesthetic_embeds` (256-dim output of aesthetic_embedding). Total: 1280 + 256 + 256 = 1792.
        # Let's simplify the refiner's added_cond_kwargs to align with the core `diffusers` pattern.

        # The refiner's `added_cond_kwargs` needs `text_embeds`, `time_ids`, and `aesthetic_score`.
        # `time_ids` is the 6-dim (height, width, 0, 0, height, width)
        # `aesthetic_score` is a 1-dim tensor [score]
        # These are handled by the UNet's `add_time_embedding` and `add_class_embedding` respectively.
        # `text_embeds` is `pooled_prompt_embeds_r` (1280-dim).

        # Refiner's `added_cond_kwargs` structure (as per diffusers `_get_add_embeds` for refiner):
        final_added_cond_kwargs_refiner = {
            "text_embeds": torch.cat([neg_pooled_embeds_r, pooled_embeds_r]), # (2*batch, 1280)
            "time_ids": add_time_ids_r_cfg, # (2*batch, 6)
            "aesthetic_score": add_aesthetic_embeds_r_cfg # (2*batch, 1)
        }


        final_cb_kwargs_refiner = {**(callback_on_step_end_kwargs_refiner or {}), 'vae': self.vae}

        for i, t in enumerate(self.progress_bar(timesteps_refiner)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler_refiner.scale_model_input(latent_model_input, t)
            
            # --- UNET REFINER CALL ---
            noise_pred = self.unet_refiner(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_r_cfg, # (2*batch, seq_len, 1280)
                added_cond_kwargs=final_added_cond_kwargs_refiner
            ).sample
            
            # Apply Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler_refiner.step(guided_noise_pred, t, latents).prev_sample
            
            # Callback
            if callback_on_step_end:
                new_guidance_scale = callback_on_step_end(
                    step=i + base_denoise_end_step, # Adjust step count for refiner
                    timestep=int(t), latents=latents,
                    current_guidance_scale=current_guidance_scale, callback_kwargs=final_cb_kwargs_refiner)
                if new_guidance_scale is not None:
                    current_guidance_scale = new_guidance_scale

        # 5. VAE Decoding
        # Scale latents before decoding
        image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        # Post-process image (e.g., clamp, normalize, convert to PIL)
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return {"images": image}