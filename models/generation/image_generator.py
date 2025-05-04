import torch
import os
import asyncio
from PIL import Image
from typing import Optional, Union, List
from diffusers import KandinskyPipeline, StableDiffusionPipeline


class DiffusionGenerator:
    def __init__(self, model_type="kandinsky"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        if model_type == "kandinsky":
            self.pipeline = KandinskyPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)

    async def generate(
        self, 
        prompt: str, 
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None
    ) -> Image.Image:
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        if negative_prompt is None:
            negative_prompt = "low quality, bad anatomy, blurry, pixelated, watermark, signature, text"
        
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.no_grad():
                if self.model_type == "kandinsky":
                    output = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        width=width,
                        height=height,
                        generator=generator
                    )
                else:
                    output = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        width=width,
                        height=height,
                        generator=generator
                    )
                
                return output.images[0]
        
        result = await loop.run_in_executor(None, _generate)
        
        return result