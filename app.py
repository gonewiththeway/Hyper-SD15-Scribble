import spaces
import argparse
import os
import time
from os import path
from PIL import ImageOps

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

import gradio as gr
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from scheduling_tcd import TCDScheduler

torch.backends.cuda.matmul.allow_tf32 = True

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")

if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")
pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SD15-1step-lora.safetensors", adapter_name="default")
pipe.to("cuda")
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config, timestep_spacing ="trailing")

with gr.Blocks(js=js_func) as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                # scribble = gr.Image(source="canvas", tool="color-sketch", shape=(512, 512), height=768, width=768, type="pil")
                scribble = gr.ImageEditor(type="pil", image_mode="L", crop_size=(512, 512), sources=(), brush=gr.Brush(color_mode="fixed", colors=["#FFFFFF"]), canvas_size=(1024, 1024))
                # scribble_out = gr.Image(height=384, width=384)
                num_images = gr.Slider(label="Number of Images", minimum=1, maximum=8, step=1, value=4, interactive=True)
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=8, step=1, value=1, interactive=True)
                prompt = gr.Text(label="Prompt", value="a photo of a cat", interactive=True)
                eta = gr.Number(label="Eta (Corresponds to parameter eta (η) in the DDIM paper, i.e. 0.0 eqauls DDIM, 1.0 equals LCM)", value=1., interactive=True)
                controlnet_scale = gr.Number(label="ControlNet Conditioning Scale", value=1.0, interactive=True)
                seed = gr.Number(label="Seed", value=3413, interactive=True)
                btn = gr.Button(value="run")

            with gr.Column():
                output = gr.Gallery(height=768, format="png")
                # output = gr.Image()

        @spaces.GPU
        def process_image(steps, prompt, controlnet_scale, eta, seed, scribble, num_images):
            global pipe
            if scribble:                
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), timer("inference"):
                    result = pipe(
                        prompt=[prompt]*num_images,
                        image=[ImageOps.invert(scribble['composite'])]*num_images,
                        # image=[scribble['composite']]*num_images,
                        generator=torch.Generator().manual_seed(int(seed)),
                        num_inference_steps=steps,
                        guidance_scale=0.,
                        eta=eta,
                        controlnet_conditioning_scale=float(controlnet_scale),
                    ).images
                    # result[0].save("test.jpg")
                    # print(result[0])
                    return result
            else:
                return None

        reactive_controls = [steps, prompt, controlnet_scale, eta, seed, scribble, num_images]

        for control in reactive_controls:
            if reactive_controls[-2] is not None:
                control.change(fn=process_image, inputs=reactive_controls, outputs=[output, ])

        btn.click(process_image, inputs=reactive_controls, outputs=[output, ])

if __name__ == "__main__":
    demo.launch()