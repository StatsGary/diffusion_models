from dreambooth.dataloader import pull_dataset_from_hf_hub, DreamBoothDataset
from dreambooth.image import image_grid
from dreambooth.collator import collate_fn
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel
from argparse import Namespace

# Project constants / variables
HF_DATA_LOCATION = 'StatsGary/dreambooth-hackathon-images'
STABLE_DIFFUSION_NAME = "CompVis/stable-diffusion-v1-4"
FEATURE_EXTRACTOR = "openai/clip-vit-base-patch32"
learning_rate = 2e-06
max_train_steps = 400
resolution=512
train_batch_size=1
grad_accum_steps=8
max_gradient_norm=1.0
sample_batch_size=2

if __name__ =='__main__':
    # Load the image dataset from HuggingFace hub
    dataset = pull_dataset_from_hf_hub(dataset_id=HF_DATA_LOCATION)
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
    # Create training arguments
    args = Namespace(
        pretrained_model_name_or_path=model_id,
        resolution=resolution, 
        train_dataset=train_dataset,
        instance_prompt=instance_prompt,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum_steps, 
        max_grad_norm=max_gradient_norm,
        gradient_checkpointing=True,  
        use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
        seed=3434554,
        sample_batch_size=sample_batch_size, #Normally 2
        output_dir="norweigen-fjords-dreambooth" 
    )