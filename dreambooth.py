
import argparse
import itertools
import math
import os
from contextlib import nullcontext
import random
import gc
import Path
from argparse import Namespace
from Utils import Utils
import accelerate

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import requests
import PIL
from io import BytesIO
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import DPMSolverMultistepScheduler

import bitsandbytes as bnb

pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5" #@param ["stabilityai/stable-diffusion-2", "stabilityai/stable-diffusion-2-base", "CompVis/stable-diffusion-v1-4", "stable-diffusion-v1-5/stable-diffusion-v1-5"] {allow-input: true}



"""
This Class is the representation of the Dreambooth Finetuning pipeline
We use a stable diffusion model as a base and we have images on top of it with this pipeline
"""
class DreamBooth():
    def __init__(self,
                 save_path="/my_concept",
                 prior_preservation=False,
                 instance_prompt="<trt-cttc> soldier",
                 prior_preservation_class_prompt="a photo of a soldier",
                 num_class_images=12,
                 sample_batch_size=2,
                 prior_loss_weight=0.5,
                 prior_preservation_class_folder="/class_images",
                 prompt="<trt-cttc> spanish cyborg",
                 num_samples=4,
                 num_rows=4
                 ):
        self.prompt = prompt
        self.num_samples = num_samples
        self.num_rows = num_rows
        self.save_path = save_path
        self.prior_preservation = prior_preservation
        self.instance_prompt = instance_prompt
        self.prior_preservation_class_prompt = prior_preservation_class_prompt
        self.num_class_images = num_class_images
        self.sample_batch_size = sample_batch_size
        self.prior_loss_weight = prior_loss_weight
        self.class_data_root = prior_preservation_class_folder
        self.class_prompt = prior_preservation_class_prompt
        self.prior_preservation_class_folder=prior_preservation_class_folder
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.tokenizer = None
        self.args = None
        self.pipe = None

    def class_images(self):
        """
        this is to generate class images if the prior_preservation parameter is true
        :return:
        """
        if self.prior_preservation:
            class_images_dir = Path(self.class_data_root)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images[0]:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path, revision="fp16", torch_dtype=torch.float16
                ).to("cuda")
                pipeline.enable_attention_slicing()
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images[0] - cur_class_images
                print(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.sample_batch_size[0])

                for example in tqdm(sample_dataloader, desc="Generating class images"):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
                pipeline = None
                gc.collect()
                del pipeline
                with torch.no_grad():
                    torch.cuda.empty_cache()

    def load_model(self):
        """
        this method is used to get the differents values from our pretrained model we use
        we'll then use them later on in the training
        :return:
        """
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

    def setting_up_args(self):
        """
        the setting_up_args() method is used to generate arguments for the training loop
        it create parameters such as the learning_rate and the train_batch_size
        :return:
        """
        self.args = Namespace(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            resolution=self.vae.sample_size,
            center_crop=True,
            train_text_encoder=False,
            instance_data_dir=self.save_path,
            instance_prompt=self.instance_prompt,
            learning_rate=5e-06,
            max_train_steps=400,
            save_steps=300,
            train_batch_size=2,  # set to 1 if using prior preservation
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            mixed_precision="fp16",  # set to "fp16" for mixed-precision training.
            gradient_checkpointing=True,  # set this to True to lower the memory usage.
            use_8bit_adam=True,  # use 8bit optimizer from bitsandbytes
            seed=3434554,
            with_prior_preservation=self.prior_preservation,
            prior_loss_weight=self.prior_loss_weight,
            sample_batch_size=2,
            class_data_dir=self.prior_preservation_class_folder,
            class_prompt=self.prior_preservation_class_prompt,
            num_class_images=self.num_class_images,
            lr_scheduler="constant",
            lr_warmup_steps=100,
            output_dir="dreambooth-concept-cat",
        )

    def training_function(self):
        """
        this is the definition of the training loop we'll use later
        :return:
        """
        logger = get_logger(__name__)

        set_seed(self.args.seed)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
        )

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
        if self.args.train_text_encoder and self.args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        self.vae.requires_grad_(False)
        if not self.args.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.args.use_8bit_adam:
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (
            itertools.chain(self.unet.parameters(),
                            self.text_encoder.parameters()) if self.args.train_text_encoder else self.unet.parameters()
        )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
        )

        noise_scheduler = DDPMScheduler.from_config(self.args.pretrained_model_name_or_path, subfolder="scheduler")

        train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            instance_prompt=self.args.instance_prompt,
            class_data_root=self.args.class_data_dir if self.args.with_prior_preservation else None,
            class_prompt=self.args.class_prompt,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # concat class and instance examples for prior preservation
            if self.args.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
            ).input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=collate_fn
        )

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        if self.args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.unet, optimizer, train_dataloader, lr_scheduler
            )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.vae.decoder.to("cpu")
        if not self.args.train_text_encoder:
            self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = self.args.train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for _ in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if self.args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), self.text_encoder.parameters())
                            if self.args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(unet.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.save_steps == 0:
                        if accelerator.is_main_process:
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                self.args.pretrained_model_name_or_path,
                                unet=accelerator.unwrap_model(unet),
                                text_encoder=accelerator.unwrap_model(self.text_encoder),
                            )
                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            pipeline.save_pretrained(save_path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= self.args.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(self.text_encoder),
            )
            pipeline.save_pretrained(self.args.output_dir)

    def training(self):
        """
        this method is the training part of the class, it is about the training of the stable diffusion
        model on our images.
        :return:
        """
        accelerate.notebook_launcher(self.training_function, args=(self.text_encoder, self.vae, self.unet), num_processes=1)
        for param in itertools.chain(self.unet.parameters(), self.text_encoder.parameters()):
            if param.grad is not None:
                del param.grad  # free some memory
            torch.cuda.empty_cache()

    def set_pipeline(self):
        """
        this is used to set the pipeline used later in the generation of the image
        if it does not exist already it makes a new one based on the training we just did
        :return:
        """
        try:
            self.pipe
        except NameError:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.args.output_dir,
                scheduler=DPMSolverMultistepScheduler.from_pretrained(self.args.output_dir, subfolder="scheduler"),
                torch_dtype=torch.float16,
            ).to("cuda")


    def run(self):
        """
        this method is to run the fine-tuned model on a prompt given by the user.
        it prints out the grid of images made in the end.
        :return:
        """

        all_images = []
        for _ in range(self.num_rows):
            images = self.pipe(self.prompt, num_images_per_prompt=self.num_samples, num_inference_steps=100, guidance_scale=9,
                          negative_prompt="game interface, real flags").images
            all_images.extend(images)

        grid = Utils.image_grid(all_images, self.num_rows, self.num_samples)
        print(grid)

    def go(self):
        """
        this method is used to make the class work, it calls every method in the right order
        it gets images from a github to train the model to reproduce the style of helldiver.
        :return: NOTHING
        """
        urls = [
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/1708423785-9799-capture-d-ecran_1.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/AUT_Berserker_Renders_Thumbnail_2.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/AUT_Raiders_Renders_Thumbnail_4.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/8ks3tuc9a0ic1-ezgif.com-webp-to-png-converter.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/Automatons_1.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/hq720.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/hq720.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/4142936-helldivers.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/mqdefault.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/helldivers-2s-big-new-patch-introduces-wave-of-updates-and-a_xmm2.png",
            "https://raw.githubusercontent.com/ArsGoe/onlinestock/main/Helldivers-2-Automaton-Hulk_1.png"
        ]

        images = list(filter(None, [Utils.download_image(url) for url in urls]))
        save_path = "./my_concept"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        [image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]
        Utils.image_grid(images, 1, len(images))

        self.class_images()
        self.load_model()
        self.setting_up_args()
        self.training_function()
        self.training()
        self.set_pipeline()
        self.run()

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
