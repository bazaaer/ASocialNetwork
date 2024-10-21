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

import torch
import requests
import PIL
from io import BytesIO
from matplotlib import pyplot as plt
import cv2                                  # NO FUCKING CLUE WHAT DOES THIS DO
import base64, os
from IPython.display import HTML, Image
from google.colab.output import eval_js     # AH SHIT GOTTA SWITCH THAT
from base64 import b64decode
import numpy as np
import shutil
from time import sleep
import time
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def get_requirements():
    """
    downloads all the necessary modules for the use of this file's functions
    :return:None
    """
    """
    !pip install -Uq diffusers==0.30.0 ftfy accelerate
    !pip install -Uq transformers==4.45.1
    !pip install mmengine # for the PowerPaint tool
    !pip install git-lfs
    """

def get_device():
    """
    getter for the device used by the machine. Works only with Nvidia GPUs (aka CUDA)
    :return:str the device used by the machine
    """

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    return device

def get_pipe():
    """
    getter for the inpainting pipe (the thing that generates the images)
    if the pipe has already been initialised somewhere, it returns it. Otherwise, it initialises it and returns it afterwards.
    :return:StableDiffusionInpaintPipeline the inpainting pipe
    """
    global inpainting_pipe
    try :
        inpainting_pipe
    except :
        from diffusers import StableDiffusionInpaintPipeline

        inpaiting_pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
        inpaiting_pipe = inpaiting_pipe.to(get_device())

def download_image(url):
    """
    downloads and return the image associated to the given URL
    :param url:str the URL of the desired image
    :return:PIL.Image the image using the PIL module
    """
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def image_grid(imgs, rows, cols):
    """
    used to show multiple images in an organized fashion, but arranging them inside a grid.
    :param imgs:list[PIL.Image] a list of the images needing to be shown
    :param rows:int the number of rows in the grid
    :param cols:int the number of columns in the grid
    :return:PIL.Image the grid made up of the images
    """
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def generate_mask_image(init_image, mask_prompt, out_fpath, debug=False):
    """

    :param init_image:PIL.Image the image inputed by the user
    :param mask_prompt:list[str] the user prompt to used to identify the mask (at least 2 words)
    :param out_fpath:str the name of the file that contains the mask
    :param debug:bool whether or not to enable debug mode
    :return:PIL.Image the mask for the image modification
    """
    assert len(mask_prompt) > 1 , "ERR : mask_prompt maust contain at least 2 keywords"
    try:
        global processor
        processor
    except :
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    try:
        global model
        model
    except:
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

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

    # visualize prediction
    if debug:
        _, ax = plt.subplots(1, 5, figsize=(15, len(mask_prompt) + 1))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(init_image)
        print(torch.sigmoid(preds[0][0]).shape)
        [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(mask_prompt))];
        [ax[i + 1].text(0, -15, prompts[i]) for i in range(len(mask_prompt))];

    plt.imsave(temp_fpath, torch.sigmoid(preds[1][0]))
    img2 = cv2.imread(temp_fpath)
    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # fix color format
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

    mask_image = PIL.Image.fromarray(bw_image)
    mask_image.save(out_fpath)
    return mask_image

def task_to_prompt(control_type):
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
def predict(
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
    :param pipe:StableDiffusionInpaintingPipe the pipe used for image generation
    :param input_image:PIL.Image the image that needs modifications
    :param prompt:str the user prompt
    :param fitting_degree:float how much the result should fit the demand
    :param ddim_steps:int how many iterations of image generation need to be done. More means better results, but longer execution.
    :param scale:float the creative scale and how much the result should resemble the original image. More means less uniqueness, but closer results.
    :param negative_prompt:list[str] what the result shoudl NOT include.
    :param task:str what guides the modifications (IE text guided, image guided...)
    :return:PIL.Image the same type of image that what was given
    """
    promptA, promptB, negative_promptA, negative_promptB = task_to_prompt(task)
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


def object_removal_with_instruct_inpainting(pipe, init_image, mask_image, negative_prompt, fitting_degree=1, \
                                          num_inference_steps=50, guidance_scale=12):
    """
    calls the predict method to remove a given item from an image
    :param pipe:StableDiffusionInpaintingPipe the model that modifies the image
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
    image = predict(
        pipe,
        input_image,
        "empty scene blur", # prompt
        fitting_degree,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        "object-removal" # task
    )
    return image

def object_addition_with_instruct_inpainting(pipe, init_image, mask_image, prompt, fitting_degree=1, \
                                          num_inference_steps=50, guidance_scale=12):
    """
        calls the predict method to add or modifiy a given item from an image
        :param pipe:StableDiffusionInpaintingPipe the model that modifies the image
        :param init_image:PIL.Image the starting image
        :param mask_image:PIL.Image the mask image that indicates where the modifications need to be
        :param negative_prompt:list[str] what the result should NOT contain
        :param fitting_degree:float how much the result should fit the demand
        :param num_inference_steps:int how many iterations of image generation need to be done. More means better results, but longer execution.
        :param guidance_scale:float the creative scale and how much the result should resemble the original image. More means less uniqueness, but closer results.
        :return:PIL.Image the modified image
        """
    input_image = {"image": init_image, "mask": mask_image}
    image = predict(
        pipe,
        input_image,
        prompt,
        fitting_degree,
        num_inference_steps,
        guidance_scale,
        "", # negative prompt
        "text-guided" # task
    )
    return image