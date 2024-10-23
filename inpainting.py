"""
@author Timéo Quehen, 21-10-2024

This python allows for image modification using a pretrained inpainting pipe from StableDiffusion.
Inside, a function to initialise the pipe (or get it if it already is)      - get_pipe()
a function to download the required modules                                 - get_requirements()
a function to draw a mask on a given image                                  - draw_mask()
a function to generate a mask when given a prompt                           - generate_mask()
a function to remove items from an image                                    - inpainting_add()
a function to add / modify items from an image                              - inpainting_remove()

this file is part of the project SAE S5.A.01, and is property of the IUT of Lens, France.
"""

from io import BytesIO
from matplotlib import pyplot as plt
import base64, os, cv2, io, torch, requests, PIL
from IPython.display import HTML, Image
from base64 import b64decode
import numpy as np
import shutil
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def init_brushnet_pipe():
    """
    Initilialises the pipe using the brushnet model
    :return:StableDiffusionPowerPaintBrushNetPipeline the pipe for image modification
    """
    import sys
    import os
    sys.path.insert(0, os.path.join("/content/PowerPaint"))

    from safetensors.torch import load_model, save_model
    import numpy as np
    from PIL import Image, ImageOps
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers.utils import load_image
    from diffusers import DPMSolverMultistepScheduler

    from powerpaint.models.BrushNet_CA import BrushNetModel
    from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
        StableDiffusionPowerPaintBrushNetPipeline,
    )
    # from powerpaint.power_paint_tokenizer import PowerPaintTokenizer
    from powerpaint.models.unet_2d_condition import UNet2DConditionModel
    from powerpaint.utils.utils import TokenizerWrapper, add_tokens
    from diffusers import UniPCMultistepScheduler

    checkpoint_dir = "/content/checkpoints"
    local_files_only = True

    # brushnet-based version
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        revision=None,
        torch_dtype=torch.float16,
        local_files_only=False,
    )
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="text_encoder",
        revision=None,
        torch_dtype=torch.float16,
        local_files_only=False,
    )
    brushnet = BrushNetModel.from_unet(unet)
    base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        safety_checker=None,
    )
    pipe.unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        revision=None,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
    )
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=base_model_path,
        subfolder="tokenizer",
        revision=None,
        torch_type=torch.float16,
        local_files_only=local_files_only,
    )

    # add learned task tokens into the tokenizer
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    load_model(
        pipe.brushnet,
        os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
    )

    pipe.text_encoder_brushnet.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()
    pipe = pipe.to("cuda")

    return pipe

def get_processor():
    """
    getter for the mask maker processor (to automatically generate masks)
    initialises the processor if it has not been yet
    :return:CLIPSegProcessor the processor
    """
    try:
        global processor
        processor
    except :
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    return processor

def get_mask_making_model():
    """
    getter for the mask maker model (to automatically generate masks)
    initialises the model if it has not been yet
    :return:CLIPSegForImageSegmentation the model
    """
    try:
        global model
        model
    except:
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return model

#================================================================================================================================
#================================================================================================================================
#================================================================================================================================

class InpaintingModel():

    def __init__(self):
        self.pipe = None,
        self.mask_processor = None,
        self.mask_model = None,

    def get_pipe(self):
        """
        getter for the inpainting pipe (Brushnet variant) (the thing that generates the images)
        if the pipe has not already been initialised before, it does it.
        :return:StableDiffusionPowerPaintBrushNetPipeline the inpainting pipe
        """
        if self.pipe is None:
            self.pipe = init_brushnet_pipe()
        return self.pipe()

    def task_to_prompt(self,control_type):
        """
        defines the type of generation needed for the prompt
        :param control_type:str the command inside the prompt
        :return:str,str,str,str 4 prompts that will influence the image generation
        """
        if control_type == "object-removal":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"
        elif control_type == "context-aware":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = ""
            negative_promptB = ""
        elif control_type == "shape-guided":
            promptA = "P_shape"
            promptB = "P_ctxt"
            negative_promptA = "P_shape"
            negative_promptB = "P_ctxt"
        elif control_type == "image-outpainting":
            promptA = "P_ctxt"
            promptB = "P_ctxt"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"
        else:
            promptA = "P_obj"
            promptB = "P_obj"
            negative_promptA = "P_obj"
            negative_promptB = "P_obj"

        return promptA, promptB, negative_promptA, negative_promptB

    @torch.inference_mode()
    def predict(self,
            pipe,
            input_image,
            prompt,
            fitting_degree,
            ddim_steps,
            scale,
            negative_prompt,
            task,
    ):
        """
        function that will guess what the user desires through their prompt, to make sure it stays true
        :param pipe:StableDiffusionPowerPaintBrushNetPipeline the pipe used for image generation
        :param input_image:PIL.Image the image that needs modifications
        :param prompt:str the user prompt
        :param fitting_degree:float how much the result should fit the demand
        :param ddim_steps:int how many iterations of image generation need to be done. More means better results, but longer execution.
        :param scale:float the creative scale and how much the result should resemble the original image. More means less uniqueness, but closer results.
        :param negative_prompt:list[str] what the result shoudl NOT include.
        :param task:str what guides the modifications (IE text guided, image guided...)
        :return:PIL.Image the same type of image that what was given
        """
        promptA, promptB, negative_promptA, negative_promptB = self.task_to_prompt(task)
        print(task, promptA, promptB, negative_promptA, negative_promptB)
        img = np.array(input_image["image"].convert("RGB"))

        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))

        np_inpimg = np.array(input_image["image"])
        np_inmask = np.array(input_image["mask"]) / 255.0

        np_inpimg = np_inpimg * (1 - np_inmask)

        input_image["image"] = PIL.Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

        result = pipe(
            promptA=promptA,
            promptB=promptB,
            promptU=prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            num_inference_steps=ddim_steps,
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=negative_prompt,
            guidance_scale=scale,
            width=H,
            height=W,
        ).images[0]
        return result

    def object_removal_with_instruct_inpainting(self,pipe, init_image, mask_image, negative_prompt, fitting_degree=1, \
                                                num_inference_steps=50, guidance_scale=12):
        """
        calls the predict method to remove a given item from an image
        :param pipe:StableDiffusionPowerPaintBrushNetPipeline the model that modifies the image
        :param init_image:PIL.Image the starting image
        :param mask_image:PIL.Image the mask image that indicates where the modifications need to be
        :param negative_prompt:list[str] what the result should NOT contain
        :param fitting_degree:float how much the result should fit the demand
        :param num_inference_steps:int how many iterations of image generation need to be done. More means better results, but longer execution.
        :param guidance_scale:float the creative scale and how much the result should resemble the original image. More means less uniqueness, but closer results.
        :return:PIL.Image the modified image
        """
        negative_prompt = negative_prompt + ", out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
        input_image = {"image": init_image, "mask": mask_image}
        image = self.predict(
            pipe,
            input_image,
            "empty scene blur",  # prompt
            fitting_degree,
            num_inference_steps,
            guidance_scale,
            negative_prompt,
            "object-removal"  # task
        )
        return image

    def object_addition_with_instruct_inpainting(self,pipe, init_image, mask_image, prompt, fitting_degree=1, \
                                                 num_inference_steps=50, guidance_scale=12):
        """
            calls the predict method to add or modifiy a given item from an image
            :param pipe:StableDiffusionPowerPaintBrushNetPipeline the model that modifies the image
            :param init_image:PIL.Image the starting image
            :param mask_image:PIL.Image the mask image that indicates where the modifications need to be
            :param negative_prompt:list[str] what the result should NOT contain
            :param fitting_degree:float how much the result should fit the demand
            :param num_inference_steps:int how many iterations of image generation need to be done. More means better results, but longer execution.
            :param guidance_scale:float the creative scale and how much the result should resemble the original image. More means less uniqueness, but closer results.
            :return:PIL.Image the modified image
            """
        input_image = {"image": init_image, "mask": mask_image}
        image = self.predict(
            pipe,
            input_image,
            prompt,
            fitting_degree,
            num_inference_steps,
            guidance_scale,
            "",  # negative prompt
            "text-guided"  # task
        )
        return image

    def inpaints_image(self,prompt, init_image_data, mask_data, out_file_name):
        """
        modifies elements inside the 'init_image' based on the given prompt, layer mask data, and saves it in the file 'out_file_name'
        :param prompt:str the prompt for desired changes
        :param init_image_data:binary an image encoded in binary
        :param mask_data:binary a layer mask encoded in binary
        :param out_file_name:str the name of the file in which we store the image
        :return:utf-8 an image encoded in UTF-8
        """

        binary = b64decode(mask_data.split(',')[1])
        mask_image = PIL.Image.open(io.BytesIO(binary))
        mask_image.save("mask1.png")
        with_mask = np.array(plt.imread("mask1.png")[:, :, :3])

        mask = (with_mask[:, :, 0] == 1) * (with_mask[:, :, 1] == 0) * (with_mask[:, :, 2] == 0)
        plt.imshow(mask, cmap='gray')
        plt.imsave("mask2.png", mask, cmap='gray')
        mask_image = PIL.Image.open("mask2.png")
        print("INFO : Mask obtained")

        binary = b64decode(init_image_data.split(',')[1])
        image = PIL.Image.open(io.BytesIO(binary))

        if is_empty_mask(mask_image):
            print('coucou')
            mask_image = generate_mask_image(image, prompt, "mask.png")

        split_prompt = prompt.lower().split(" ")
        instruction = "add"
        for word in split_prompt:
            if word == "remove" or word == "delete" or word == "cut":
                instruction = "remove"
                break

        if instruction == "remove":
            image = object_removal_with_instruct_inpainting(self.get_pipe(), image, mask_image.convert("RGB"), prompt)
        else:
            image = object_addition_with_instruct_inpainting(self.get_pipe(), image, mask_image.convert("RGB"), prompt)

        image.save(out_file_name)
        encoded = base64.b64encode(open(out_file_name, "rb").read()).decode('utf-8')
        print("INFO : image created and saved under the name", out_file_name)
        return encoded

    # ====================================
    #           MASK GENERATION
    # ====================================

    def is_empty_mask(self,mask_image):
        """
        checks if the given image is only black pixels (meaning the mask is empty)
        :param mask_image:PILImage the given mask image
        :return:bool whether or not the mask is empty
        """
        return cv2.countNonZero(mask_image) == 0

    def generate_mask_image(self,init_image, mask_prompt, out_fpath):
        """
        generates a mask automatically based on the given 'mask_prompt' and returns it
        :param init_image:PIL.Image the image inputed by the user
        :param mask_prompt:list[str] the user prompt to used to identify the mask (at least 2 words)
        :param out_fpath:str the name of the file that contains the mask
        :return:PIL.Image the mask for the image modification
        """
        assert len(mask_prompt) > 1, "ERR : mask_prompt maust contain at least 2 keywords"

        processor = get_processor()

        model = get_mask_making_model()

        temp_fpath = f"temp.png"
        if isinstance(mask_prompt, str):
            mask_prompt = [mask_prompt, mask_prompt]
        if isinstance(mask_prompt, list) and len(mask_prompt) == 1:
            mask_prompt = mask_prompt * 2
        inputs = processor(text=mask_prompt, images=[init_image] * len(mask_prompt), padding="max_length",
                           return_tensors="pt")

        # predict
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.unsqueeze(1)

        plt.imsave(temp_fpath, torch.sigmoid(preds[1][0]))
        img2 = cv2.imread(temp_fpath)
        gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

        # fix color format
        cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

        mask_image = PIL.Image.fromarray(bw_image)
        mask_image.save(out_fpath)
        return mask_image

