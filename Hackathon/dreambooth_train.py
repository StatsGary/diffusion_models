from dreambooth.dataloader import pull_dataset_from_hf_hub, DreamBoothDataset
from dreambooth.image import image_grid
from dreambooth.collator import collate_fn
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel
from argparse import Namespace
import math
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
import torch

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
        sample_batch_size=sample_batch_size, 
        output_dir=model_checkpoint_name
    )

    #Define training loop
    def training_function(text_encoder, vae, unet, train_dataset, train_batch_size=1, shuffle_train=True,
                          beta_start=0.00085, beta_end=0.012, beta_scheduler="scaled_linear", num_train_timesteps=1000, seed=3434554,
                          gradient_checkpoint=True, use_8bit_ADAM=True, learning_rate=2e-06,
                          ):
        # Takes the input from the training arguments to specify the warmup phase of the gradients
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        # Sets a reproduable seed to work 
        set_seed(seed)
        if gradient_checkpoint:
            unet.enable_gradient_checkpointing()

        if use_8bit_ADAM:
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Then we implemenet and optimizer class which is used with the learning rate
        optimizer = optimizer_class(
            unet.parameters(),  # only optimize unet
            lr=learning_rate,
        )
        # Create a random noise scheduler to be applied to the images
        noise_scheduler = DDPMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_scheduler,
            num_train_timesteps=num_train_timesteps
        )

        # Pass the images into the training data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn
        )

        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
        )

        # Move text_encode and vae to gpu
        text_encoder.to(accelerator.device)
        vae.to(accelerator.device)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = (
            args.train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")
        global_step = 0

        # Set the training loop for each epoch
        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                        latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample
                    loss = (
                        F.mse_loss(noise_pred, noise, reduction="none")
                        .mean([1, 2, 3])
                        .mean()
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            print(f"Loading pipeline and saving to {args.output_dir}...")
            scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
                steps_offset=1,
            )
            pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=accelerator.unwrap_model(unet),
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker"
                ),
                feature_extractor=feature_extractor,
            )
            pipeline.save_pretrained(args.output_dir)