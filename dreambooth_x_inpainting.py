import os
import shutil
import subprocess

subprocess.run("gdown 1CC0YdBBPjqy1dyW6kNfH5qwmLJnlXGhR")
subprocess.run("pip install -r ./dreambooth_inpainting_requirements.txt", shell=True)

from diffusers import StableDiffusionInpaintPipeline
import torch
from Utils import Utils

IMAGES = [
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_1.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_2.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_3.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_4.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_eagle.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_intro.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_pose.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_salute.png"),
    Utils.download_image("https://raw.githubusercontent.com/ArsGoe/onlinestock/dreambooth_x_inpainting/helldivers_thumb.png")
]

def training(images=IMAGES, model_name="runwayml/stable-diffusion-inpainting", class_name="helldiver"):
    Utils.store_images(images, "./content/dreambooth_inpainting_in")
    out_images_path = f"./content/dreambooth_inpainting_out_{class_name}"
    if os.path.exists(out_images_path):
        shutil.rmtree(out_images_path)
    os.makedirs(out_images_path)
    class_images_path = f"./content/dreambooth_inpainting_class_images_{class_name}"
    if os.path.exists(class_images_path):
        shutil.rmtree(class_images_path)
    os.makedirs(class_images_path)
    command = "accelerate launch train_dreambooth_inpaint.py " \
            + f"--pretrained_model_name_or_path={model_name} " \
            + "--instance_data_dir=./content/dreambooth_inpainting_in " \
            + f"--class_data_dir={class_images_path} " \
            + f"--output_dir={out_images_path} " \
            + "--with_prior_preservation " \
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
            + "--max_train_steps=400 " \
            + "--train_text_encoder " \
            + "--checkpointing_steps=4000"
    subprocess.run(command, shell=True)
    return StableDiffusionInpaintPipeline.from_pretrained(
            out_images_path,
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

    def generate_image_dreambooth_x_inpainting(self, prompt, init_image, mask_image, num_inference_steps=150):
        if f"a photo of dpf {self.class_name}" not in prompt:
            prompt = f"a photo of dpf {self.class_name} {prompt}"
        image = self.pipeline(prompt, image=init_image, mask_image=mask_image, num_inference_steps=num_inference_steps).images
        return image
