import torch
import os
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class FlowerArrangementGenerator:
    
    # def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", 
    #              device: str = None, output_dir: str = "data/generated"):
    def __init__(self, model_id: str = 'GSV1510/sd-flower-diffusion-32px', 
                 device: str = None, output_dir: str = "data/generated"):
       
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None  # 안전 검사기 비활성화 (꽃 이미지는 문제 없음)
        )

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print(f"Stable Diffusion model loaded on {self.device}")
    
    def generate_prompt(self, main_flower: Dict, 
                        medium_flowers: List[Dict], 
                        small_flowers: List[Dict], 
                        style: str = "natural") -> str:
        """
        Args:
            main_flower: 메인 꽃 정보
            medium_flowers: 중형 꽃 정보 목록
            small_flowers: 소형 꽃 정보 목록
            style: 이미지 스타일 ("natural", "minimalist", "romantic")
        """

        main_name = main_flower["name"]
        medium_names = ", ".join([f["name"] for f in medium_flowers])
        small_names = ", ".join([f["name"] for f in small_flowers])

        main_color = main_flower["color"]

        prompt = f"A beautiful flower arrangement featuring {main_color} {main_name} as the main flower"

        if medium_names:
            prompt += f", with {medium_names} as secondary flowers"
  
        if small_names:
            prompt += f", and {small_names} as accent flowers"

        style_prompts = {
            "natural": "in a natural, organic style. Soft daylight, professional floral photography, high detail, vibrant colors, shallow depth of field",
            "minimalist": "in a minimalist, elegant style. Clean background, simple container, professional floral photography, high detail, selective focus",
            "romantic": "in a romantic, dreamy style. Soft pastel colors, dreamy lighting, professional floral photography, high detail, bokeh background"
        }

        style_prompt = style_prompts.get(style.lower(), style_prompts["natural"])
        prompt += f" {style_prompt}"
        prompt += ", 8k, high quality, photorealistic, hyperrealistic" # 품질개선 
        
        return prompt
    

    def generate_negative_prompt(self) -> str:
        return "ugly, blurry, low quality, distorted, deformed, cartoon, illustration, painting, drawing, anime, text, watermark, signature, frame, border, collage"
    

    def generate_image(self, 
                       main_flower: Dict, 
                       medium_flowers: List[Dict], 
                       small_flowers: List[Dict], 
                       style: str = "natural",
                       seed: Optional[int] = None,
                       num_inference_steps: int = 30,
                       guidance_scale: float = 7.5) -> Tuple[Image.Image, str]:

        prompt = self.generate_prompt(main_flower, medium_flowers, small_flowers, style)
        negative_prompt = self.generate_negative_prompt()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.autocast(self.device):
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        return image, prompt
    

    def save_image(self, image: Image.Image, generation_id: str) -> str:

        file_path = os.path.join(self.output_dir, f"{generation_id}.png")
        image.save(file_path)
        
        return file_path
    

    def batch_generate(self, main_flower: Dict, 
                       medium_flowers: List[Dict], 
                       small_flowers: List[Dict], 
                       style: str = "natural",
                       num_images: int = 4) -> List[Tuple[Image.Image, str]]:
        
        results = []
        
        for i in range(num_images):
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

            image, prompt = self.generate_image(
                main_flower=main_flower,
                medium_flowers=medium_flowers,
                small_flowers=small_flowers,
                style=style,
                seed=seed
            )
            
            results.append((image, prompt))
        
        return results