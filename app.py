import torch
from io import BytesIO
import base64
from diffusers import DiffusionPipeline
import PIL
import base64
import io
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import requests


class InferlessPythonModel:
    def initialize(self):
        brightness_controlnet = ControlNetModel.from_pretrained(
            "ioclab/control_v1p_sd15_brightness", torch_dtype=torch.float16
        )

        tile_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16,
            use_safetensors=False,
        )

        controller = StableDiffusionControlNetPipeline.from_pretrained(
            "Lykon/DreamShaper",
            controlnet=[brightness_controlnet, tile_controlnet],
            torch_dtype=torch.float16,
            use_safetensors=False,
            safety_checker=None,
        ).to("cuda")

        controller.enable_xformers_memory_efficient_attention()
        self.pipe = controller

    def resize_for_condition_image(self, input_image, resolution: int = 512):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=PIL.Image.LANCZOS)
        return img



    def download_image(self, url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    
    def infer(self, inputs):

        input_image_url = inputs["input_image_url"]
        input_image = self.download_image(input_image_url).resize((512, 512))

        tile_input_image = self.resize_for_condition_image(input_image)
        
        prompt = inputs["prompt"]
        image = self.pipe(
            prompt,
            image=[input_image, tile_input_image],
            height=512,
            width=512,
            num_inference_steps=100,
            controlnet_conditioning_scale=[0.45, 0.25],
            guidance_scale=9,
        )["images"][0]
        
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue())
        return {"generated_image_base64": img_str.decode("utf-8")}

    def finalize(self):
        self.pipe = None
