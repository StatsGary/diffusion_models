from dreambooth.dataloader import pull_dataset_from_hf_hub, DreamBoothDataset
from dreambooth.image import image_grid
from dreambooth.collator import collate_fn
from dreambooth.train import train_dreambooth
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel
from argparse import Namespace
import logging
logging.basicConfig(filename="log.txt", level=logging.DEBUG)

# Project constants / variables
STABLE_DIFFUSION_NAME = "CompVis/stable-diffusion-v1-4"
FEATURE_EXTRACTOR = "openai/clip-vit-base-patch32"
hf_data_location = 'StatsGary/dreambooth-hackathon-images'
learning_rate = 2e-06
max_train_steps = 400
resolution=512
train_batch_size=1
grad_accum_steps=8
max_gradient_norm=1.0
sample_batch_size=2
model_checkpoint_name = "norweigen-fjords-dreambooth" 

if __name__ =='__main__':
    # Load the image dataset from HuggingFace hub
    dataset = pull_dataset_from_hf_hub(dataset_id=hf_data_location)

    # Name your concept and set of images
    name_of_your_concept = "norweigen-fjords"  
    type_of_thing = "fjords"  
    instance_prompt = f"a photo of {name_of_your_concept} {type_of_thing}"
    print(f"Instance prompt: {instance_prompt}")

    # Load the CLIP tokenizer
    model_id = STABLE_DIFFUSION_NAME
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer")

    # Create a train dataset from the Dreambooth data loader
    train_dataset = DreamBoothDataset(dataset, instance_prompt, tokenizer)

    # Get text encoder, UNET and VAE
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(FEATURE_EXTRACTOR)

    # Train the model
    model = train_dreambooth(
        text_encoder=text_encoder, 
        vae = vae, 
        unet = unet, 
        tokenizer=tokenizer, 
        feature_extractor=feature_extractor, 
        train_dataset=train_dataset, 
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps, 
        shuffle_train=True,
        gradient_accumulation_steps=grad_accum_steps, 
        use_8bit_ADAM=True, 
        learning_rate=learning_rate, 
        max_grad_norm=max_gradient_norm,
        output_dir=model_checkpoint_name
    )