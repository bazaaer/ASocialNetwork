# face_swap_with_masks.py

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def generate_mask_image(image, mask_prompt, out_fpath, processor, model, debug=False):
    temp_fpath = "temp.png"
    if isinstance(mask_prompt, str):
        mask_prompt = [mask_prompt, mask_prompt]
    if isinstance(mask_prompt, list) and len(mask_prompt) == 1:
        mask_prompt = mask_prompt * 2

    # Convert the image to PIL format
    init_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    inputs = processor(text=mask_prompt, images=[init_image] * len(mask_prompt), return_tensors="pt")
    # Removed 'padding' argument as it's not valid for ViTImageProcessor.preprocess

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.unsqueeze(1)

    # Visualize prediction (optional)
    if debug:
        _, ax = plt.subplots(1, len(mask_prompt) + 1, figsize=(15, len(mask_prompt) + 1))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(init_image)
        print(torch.sigmoid(preds[0][0]).shape)
        [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(mask_prompt))]
        [ax[i + 1].text(0, -15, mask_prompt[i]) for i in range(len(mask_prompt))]
        plt.savefig('debug_plot.png')  # Save the plot to a file instead of showing it
        plt.close()

    # Save the mask
    plt.imsave(temp_fpath, torch.sigmoid(preds[1][0]), cmap='gray')
    img2 = cv2.imread(temp_fpath)
    os.remove(temp_fpath)  # Clean up temporary file
    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # Convert to RGB
    bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB)

    mask_image = Image.fromarray(bw_image)
    mask_image.save(out_fpath)
    return bw_image

# Load the CLIPSeg model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load the images
soldier_img = cv2.imread('helldiver_generated_image.jpg')
my_face_img = cv2.imread('me.jpg')

if soldier_img is None:
    print("Soldier image not found.")
    exit()
if my_face_img is None:
    print("My face image not found.")
    exit()

# Generate the mask for the soldier's face (or head/helmet)
soldier_mask_prompt = "soldier's face"
soldier_mask = generate_mask_image(soldier_img, soldier_mask_prompt, 'soldier_mask.png', processor, model)

# Generate the mask for your face
my_face_mask_prompt = "face"
my_face_mask = generate_mask_image(my_face_img, my_face_mask_prompt, 'my_face_mask.png', processor, model)

# Convert masks to grayscale
soldier_mask_gray = cv2.cvtColor(soldier_mask, cv2.COLOR_BGR2GRAY)
my_face_mask_gray = cv2.cvtColor(my_face_mask, cv2.COLOR_BGR2GRAY)

# Find contours in the soldier's mask
contours, _ = cv2.findContours(soldier_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No contours found in the soldier's mask.")
    exit()
# Assume the largest contour is the face region
soldier_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(soldier_contour)

# Resize your face image and mask to match the soldier's face region
resized_face = cv2.resize(my_face_img, (w, h))
resized_face_mask = cv2.resize(my_face_mask_gray, (w, h))

# Create a composite mask for your face
_, resized_face_mask = cv2.threshold(resized_face_mask, 127, 255, cv2.THRESH_BINARY)

# Place your face onto the soldier's image using the mask
# Create an ROI in the soldier's image
roi = soldier_img[y:y+h, x:x+w]
roi_mask = soldier_mask_gray[y:y+h, x:x+w]

# Invert the soldier's mask to get the area to replace
roi_soldier_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(roi_mask))
# Apply your face mask
roi_my_face_fg = cv2.bitwise_and(resized_face, resized_face, mask=resized_face_mask)

# Combine the background and your face
dst = cv2.add(roi_soldier_bg, roi_my_face_fg)

# Place the combined image back into the soldier's image
soldier_img[y:y+h, x:x+w] = dst

# Optionally, blend the edges using seamless cloning
# Create a mask for seamless cloning
seamless_clone_mask = np.zeros_like(soldier_mask_gray)
seamless_clone_mask[y:y+h, x:x+w] = resized_face_mask

# Center point for seamless cloning
center = (x + w // 2, y + h // 2)

# Perform seamless cloning
output = cv2.seamlessClone(soldier_img, soldier_img, seamless_clone_mask, center, cv2.NORMAL_CLONE)

# Save the result
cv2.imwrite('face_swapped.jpg', output)

# # Clean up mask images
# os.remove('soldier_mask.png')
# os.remove('my_face_mask.png')

print("Face swap completed. The result is saved as 'face_swapped.jpg'.")
