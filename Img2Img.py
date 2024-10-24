import torch
from PIL import Image
from mpmath.libmp.libmpf import negative_rnd

from Utils import *
from matplotlib import pyplot as plt
import random as rdn

from diffusers import StableDiffusionImg2ImgPipeline

device = Utils.get_device()

class Img2Img:
    """
    This class will use the Stable Diffusion model to perform image to image tasks.
    """
    def __init__(self):
        """
        Initialize the model
        """
        self.model_id = "stabilityai/stable-diffusion-2-1-base"
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id, variant="fp16",
                                                                      use_safetensors=True).to(device)

    def generate_image(self, prompt: str | list[str] , input_image: Image.Image, num_image: int = 1, seed: int=None, strength: float=0.3):
        """
        Generate an image from the input image
        :param prompt: The prompt for the image
        :param input_image: The input image : Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, List[PIL.Image.Image], List[numpy.ndarray], List[torch.Tensor]]
        :param num_image: The number of images to generate
        :param seed: The seed for the image (ADN)
        :param strength: The strength of the image
        :return: The generated image
        """
        prompt = [prompt] * num_image
        input_image = [input_image] * num_image

        if seed is None:
            seed = rdn.randint(0, 1000000000)

        negative_prompt = "Fantasy elements like magic, dragons, knights in medieval armor, cartoonish or unrealistic characters. Avoid bright and cheerful settings, peaceful landscapes, or overly stylized elements. No human-like faces on robots, no friendly or harmless-looking creatures. Avoid any historical or ancient battlefields—focus on futuristic technology and alien environments."

        # Generate the image
        generated_image = self.img2img_pipe(
            prompt,
            image=input_image,
            generator=torch.Generator().manual_seed(seed),
            strength=strength,
            negative_prompt=negative_prompt
        )

        return generated_image
