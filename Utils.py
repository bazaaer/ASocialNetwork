import torch
import random
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


class Utils:
    @staticmethod
    def get_random_number():
        return random.randint(1, 10000000)

    @staticmethod
    def get_device():
        device = ("mps"
                  if torch.backends.mps.is_available()
                  else "cuda"
        if torch.cuda.is_available()
        else "cpu")
        return device

    @staticmethod
    def download_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

    @staticmethod
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    @staticmethod
    def store_images(images: list[Image.Image],dest_path:str)->bool:
        """
        Store the images in the destination path
        If the path does not exist, it will be created
        If the path is not empty, it will be cleared
        :param images:
        :param dest_path:
        :return:
        """
        import os
        import shutil

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        os.makedirs(dest_path)

        for i, img in enumerate(images):
            img.save(f"{dest_path}/{i}.png")
        return True

    @staticmethod
    def show_image(image):
        """
        Show the image.
        :param image: The image to show
        """
        plt.imshow(image)
        plt.axis('off')
        plt.show()
