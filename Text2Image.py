"""
@author Clémence Despeghel, 21-10-2024

This Python script generates images from text prompts using a pretrained StableDiffusion pipeline.

This file is part of the project SAÉ S5.A.01, and is property of the IUT of Lens, France.
"""

import torch
from diffusers import StableDiffusionPipeline


class Text2Image:
    def __init__(self):
        """
        Initialize the image generator.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def generate_image(self, prompt, styles=[], seed=1024, guidance_scale=9, num_inference_steps=50,
                       negative_prompt=""):
        """
        Generate an image from a prompt.

        Args:
            prompt (str): The main text prompt describing the image.
            styles (list, optional): Additional styles to refine the output (e.g., "black and white"). Defaults to [].
            seed (int, optional): Random seed for reproducibility. Defaults to 1024.
            guidance_scale (float, optional): Strength of adherence to the prompt. Defaults to 9.
            num_inference_steps (int, optional): Number of inference steps for generation. Defaults to 50.
            negative_prompt (str, optional): Elements to exclude from the image. Defaults to "".

        Returns:
            image: The generated image based on the input prompt and styles.
        """
        if styles != []:
            for i in range(len(styles)):
                prompt += ", " + styles[i]

        generator = torch.Generator("cuda").manual_seed(seed)
        image = self.pipe(prompt, generator=generator, guidance_scale=guidance_scale,
                          num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]

        return image
