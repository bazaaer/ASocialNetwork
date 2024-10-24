# stable_diffusion_model.py

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

class StableDiffusionModel:
    def __init__(self, use_lora=False, lora_weights_path=None, base_model="sd-legacy/stable-diffusion-v1-5"):
        """
        Initialize the Stable Diffusion model with optional LoRA weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = base_model
        self.use_lora = use_lora
        self.lora_weights_path = lora_weights_path

        # Load the base model for Text-to-Image
        self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model, torch_dtype=torch.float16
        ).to(self.device)

        # Enable memory-efficient attention if available
        try:
            self.text2img_pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory-efficient attention.")
        except Exception as e:
            print(f"Could not enable xformers attention: {e}")

        # Apply LoRA weights if specified
        if self.use_lora and self.lora_weights_path:
            try:
                self.load_lora_weights()
            except FileNotFoundError:
                print(f"LoRA weights at {self.lora_weights_path} not found.")

    def load_lora_weights(self):
        """Load LoRA weights onto the pipeline if specified."""
        print(f"Loading LoRA weights from {self.lora_weights_path}")
        self.text2img_pipe.load_lora_weights(self.lora_weights_path)

    def move_model_to_cpu(self):
        """Move the model to CPU to free up GPU memory."""
        self.text2img_pipe.to('cpu')
        self.device = 'cpu'
        torch.cuda.empty_cache()
        print("Moved model to CPU and cleared GPU cache.")

    def move_model_to_gpu(self):
        """Move the model back to GPU for inference."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text2img_pipe.to(self.device)
        print("Moved model to GPU for inference.")

    def generate_text_to_image(self, prompt, negative_prompt=None, guidance_scale=7, num_inference_steps=31, seed=None):
        """
        Generate an image from a text prompt using the Text-to-Image pipeline.
        """
        if self.device != 'cuda':
            self.move_model_to_gpu()

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")
        else:
            generator = None

        with torch.no_grad():
            image = self.text2img_pipe(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

        self.move_model_to_cpu()
        return image
