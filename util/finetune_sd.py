import os
import argparse
import json
import torch
import logging
import torch.nn.functional as F
import numpy as np 

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowerDataset(Dataset):
    def __init__(self, data_root, tokenizer, size=512, center_crop=True):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.size = size 
        self.center_crop = center_crop 
        self.image_paths = []
        self.captions = []

        for flower_dir in self.data_root.iterdir():
            if flower_dir.is_dir():
                flower_name = self._clean_flower_name(flower_dir.name)

                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
                for ext in image_extensions:
                    for img_path in flower_dir.glob(ext):
                        self.image_paths.append(img_path)
                        caption = f'a beautiful {flower_name} flower'
                        self.captions.append(caption)

        logger.info(f'Found {len(self.image_paths)} images across {len(set([p.parent.name for p in self.image_paths]))} flower types')
    
    def _clean_flower_name(self, raw_name):
        """Clean flower name by fixing broken characters and formatting"""
        cleaned = raw_name.replace('ÇÇ', "'").replace('çç', "'").replace('ççç', "'")
        # Fix other common encoding issues
        cleaned = cleaned.replace('Ã¢', "'").replace('â€™', "'").replace('â€˜', "'")
        
        # Handle camelCase by adding spaces before capital letters
        import re
        # add space before capital letters that follow lowercase letters 
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
        cleaned = cleaned.lower() 
        cleaned = re.sub(r'\s+', ' ', cleaned).strip() # clean up multiple spaces and trim 

        logger.debug(f"Cleaned flower name: '{raw_name}' -> '{cleaned}'")
        return cleaned 
    
    
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(image_path).convert("RGB")

        if self.center_crop:
            image = self._center_crop_resize(image)
        else:
            image = image.resize((self.size, self.size), Image.LANCZOS)
        
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0 
        image = image.permute(2, 0, 1)

        input_ids = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        ).input_ids[0]

        return {
            'pixel_values': image, 
            'input_ids': input_ids,
            'caption': caption 
        }
    

    def _center_crop_resize(self, image):
        w, h = image.size
        min_dim = min(w, h)

        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        image = image.crop((left, top, right, bottom))
        return image.resize((self.size, self.size), Image.LANCZOS)
    

class StableDiffusionTrainer:
    def __init__(self, args):
        self.args = args 
        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        self.load_models() 
        self.setup_dataset() 
        self.setup_optimizer() 

    def load_models(self):
        logger.info('Loading pretrained models..')
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.pretrained_model_name,
            subfolder='tokenizer'
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name,
            subfolder='text_encoder'
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name, 
            subfolder='vae'
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name, 
            subfolder='unet'
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name,
            subfolder='scheduler'
        )
        self.vae.requires_grad_(False) # freeze vae 

        if not self.args.train_text_encoder:
            self.text_encoder.requires_grad_(False)

    
    def setup_dataset(self):
        self.train_dataset = FlowerDataset(
            self.args.train_data_dir,
            self.tokenizer,
            size=self.args.resolution
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.args.train_batch_size,
            shuffle=True,
            num_workers=4
        )
    
    
    def setup_optimizer(self):
        params_to_optimize = []
        if self.args.train_text_encoder:
            params_to_optimize.extend(self.text_encoder.parameters())
        params_to_optimize.extend(self.unet.parameters())

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

    def train(self):
        logger.info("Starting training...")
        
        # Prepare for distributed training
        (
            self.unet,
            self.text_encoder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet,
            self.text_encoder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        
        self.vae.to(self.accelerator.device)
        if not self.args.train_text_encoder:
            self.text_encoder.to(self.accelerator.device)
        
        total_batch_size = (
            self.args.train_batch_size* self.accelerator.num_processes* self.args.gradient_accumulation_steps
        )
        
        logger.info(f"Num examples = {len(self.train_dataset)}")
        logger.info(f"Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps = {self.args.max_train_steps}")
        

        global_step = 0
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        
        for epoch in range(self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, 
                        (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        params_to_clip = []
                        if self.args.train_text_encoder:
                            params_to_clip.extend(self.text_encoder.parameters())
                        params_to_clip.extend(self.unet.parameters())
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update progress
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logs = {
                        "loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % self.args.checkpointing_steps == 0:
                        self.save_checkpoint(global_step)
                    
                    if global_step >= self.args.max_train_steps:
                        break
            
            if global_step >= self.args.max_train_steps:
                break

        self.save_model()
        logger.info("Training completed!")
    
    def save_checkpoint(self, step):
        if self.accelerator.is_main_process:
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            unet = self.accelerator.unwrap_model(self.unet)
            unet.save_pretrained(checkpoint_dir / "unet")

            if self.args.train_text_encoder:
                text_encoder = self.accelerator.unwrap_model(self.text_encoder)
                text_encoder.save_pretrained(checkpoint_dir / "text_encoder")
            
            logger.info(f"Saved checkpoint at step {step}")
    
    def save_model(self):
        if self.accelerator.is_main_process:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create pipeline and save
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder) if self.args.train_text_encoder else None,
            )
            pipeline.save_pretrained(output_dir)
            
            logger.info(f"Model saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on flower dataset")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to training data directory containing flower subdirectories"
    )
    
    # Training arguments
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for input images")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train text encoder")
    
    # Optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained model")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    
    # Accelerate arguments
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    
    return parser.parse_args()

def main():
    args = parse_args()

    trainer = StableDiffusionTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()

