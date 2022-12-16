from fastapi import FastAPI, HTTPException, File, UploadFile

from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    shape = [4, h // 8, w // 8]
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     start_code=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                    )
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")
    
    
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

def decode_to_im(model, samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(ims.cpu().numpy(), '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n_samples//nrow, n2=nrow)
    return Image.fromarray(x_sample.astype(np.uint8))
    


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route
@app.get("/")
def root():
    return {"message": "Welcome to Your image editing api"}


# Define the route to the sentiment predictor
@app.post("/edit_image")
def edit_image(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("uploads/{}".format(file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
    device = "cpu"
    config="/Users/sachinbonagiri/Documents/tech_work/image_editor/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    ckpt = "/Users/sachinbonagiri/Documents/tech_work/image_editor/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"
    input_image = "uploads/{}".format(file.filename)
    prompt = "A photo of Barack Obama smiling with a big grin"

    # Generation parameters
    scale=3
    h=512
    w=512
    ddim_steps=45
    ddim_eta=0.0
    torch.manual_seed(0)

    model = load_model_from_config(config, ckpt, device)
    sampler = DDIMSampler(model)

    init_image = load_img(input_image).to(device).unsqueeze(0)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    decode_to_im(model, init_latent)

    orig_emb = model.get_learned_conditioning([prompt])
    emb = orig_emb.clone()


    emb.requires_grad = True
    lr = 0.001
    it = 500
    opt = torch.optim.Adam([emb], lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    # Sample the model with a fixed code to see what it looks like
    quick_sample = lambda x, s, code: decode_to_im(sample_model(model, sampler, x, h, w, ddim_steps, s, ddim_eta, start_code=code))
    start_code = torch.randn_like(init_latent)

    quick_sample(emb, scale, start_code)

    pbar = tqdm(range(it))
    for i in pbar:
        opt.zero_grad()
        
        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(1000, (1,), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)
        
        pred_noise = model.apply_model(z, t_enc, emb)
        
        loss = criteria(pred_noise, noise)
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()



    # Interpolate the embedding
    for alpha in (0.8, 0.9, 1, 1.1):
        new_emb = alpha*orig_emb + (1-alpha)*emb
        quick_sample(new_emb, scale, start_code)

    return {"text_message": "success"}
