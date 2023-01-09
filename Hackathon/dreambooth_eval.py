""" 
Name:       dreambooth_eval.py
Author:     Gary Hutson
Date:       09/01/2023
Usage:      python dreambooth_eval.py
"""

from dreambooth.image import image_grid
from diffusers import  StableDiffusionPipeline
import torch


if __name__ == '__main__':
    # Load in fine tuned model
    model_name = 'norweigen-fjords-dreambooth'
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    # A prompt to test
    prompt = f"a photo of a viking on a fjord"
    guidance_scale = 7

    num_cols = 2
    all_images = []
    for _ in range(num_cols):
        images = pipe(prompt, guidance_scale=guidance_scale).images
        all_images.extend(images)

    plt = image_grid(all_images, 1, num_cols)
    plt.save('images/viking.jpg')