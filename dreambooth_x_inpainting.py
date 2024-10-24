import os

os.system("gdown 1CC0YdBBPjqy1dyW6kNfH5qwmLJnlXGhR")
os.system("pip install -qqU dreambooth_inpainting_requirements.txt")

from diffusers import StableDiffusionInpaintPipeline
import torch
from utils import download_image, store_images
#from accelerate import Accelerator
#from accelerate.logging import get_logger

IMAGES = [
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_1.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_2.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_3.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_4.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_eagle.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_intro.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_pose.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_salute.png"),
    download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_thumb.png")
]

def training(images=IMAGES, model_name="runwayml/stable-diffusion-inpainting", class_name="helldiver"):
    store_images(images, "./content/dreambooth_inpainting_in")
    command = "accelerate launch train_dreambooth_inpaint.py " \
            + f"--pretrained_model_name_or_path={model_name} " \
            + "--instance_data_dir=./content/dreambooth_inpainting_in " \
            + f"--class_data_dir=./content/dreambooth_inpainting_class_images_{class_name} " \
            + f"--output_dir=./content/dreambooth_inpainting_out_{class_name} " \
            + "--with_prior_preservation" \
            + "--prior_loss_weight=1.0 " \
            + f"--instance_prompt='a photo of dpf {class_name}' " \
            + f"--class_prompt='a photo of {class_name}' " \
            + "--resolution=512 " \
            + "--train_batch_size=1 " \
            + "--gradient_accumulation_steps=1 " \
            + "--use_8bit_adam " \
            + "--learning_rate=5e-6 " \
            + "--lr_scheduler='constant' " \
            + "--lr_warmup_steps=0 " \
            + "--num_class_images=200 " \
            + "--max_train_steps=1200 " \
            + "--train_text_encoder " \
            + "--checkpointing_steps=4000"
    os.system(command)
    return StableDiffusionInpaintPipeline.from_pretrained(
            "./content/dreambooth_inpainting_out_helldiver",
            torch_dtype=torch.float16
        ).to("cuda")

pipe = training()

class DreamboothXInpainting:
    def __init__(self, images=None, class_name="helldiver_2"):
        if images is None:
            self.pipeline = pipe
        else:
            if class_name == "helldiver":
                class_name = "helldiver_2"
            self.pipeline = training(images=images, class_name=class_name)
        self.class_name = class_name

    def generate_image_dreambooth_x_inpainting(self, prompt, init_image, mask_image):
        if f"a photo of dpf {self.class_name}" not in prompt:
            prompt = f"a photo of dpf {self.class_name} {prompt}"
        image = self.pipeline(prompt, image=init_image, mask_image=mask_image, num_inference_steps=150).images
        return image
