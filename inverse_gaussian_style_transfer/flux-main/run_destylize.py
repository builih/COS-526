"""
FLUX.1-dev img2img destylization: converts paintings → photorealistic images.

Uses SDEdit (encode→noise→denoise) with rectified-flow scheduling.
Strength controls how much the painting is altered:
  - 0.3 = subtle, keeps most paint texture
  - 0.5 = moderate, good balance
  - 0.7 = aggressive, very photorealistic but may lose composition detail
"""

import os, gc, glob, json, math, sys
import torch
import numpy as np
from einops import rearrange
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_flow_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOCAL_MODELS = "/scratch/gpfs/KAUSHIKS/yl4841/flux-main/local_models"


class LocalHFEmbedder(nn.Module):
    def __init__(self, path, max_length, is_clip=False, **hf_kwargs):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if is_clip else "last_hidden_state"
        if is_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(path, max_length=max_length)
            self.hf_module = CLIPTextModel.from_pretrained(path, **hf_kwargs)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(path, max_length=max_length)
            self.hf_module = T5EncoderModel.from_pretrained(path, **hf_kwargs)
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_length=False, return_overflowing_tokens=False,
            padding="max_length", return_tensors="pt",
        )
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None, output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()


PHOTO_SUFFIX = (
    ", DSLR photograph shot on Canon EOS R5 with 35mm lens, "
    "sharp focus, natural lighting, high dynamic range, 8k resolution, "
    "no painting, no illustration, no brushstrokes, real photograph"
)

DEFAULT_PROMPTS = {
    "green_land.jpg":
        "a real photograph of green rolling hills overlooking a distant lake, "
        "conifer and deciduous trees on a grassy hillside, clear blue sky with soft clouds"
        + PHOTO_SUFFIX,
    "robert-julian-onderdonk_texas-dry-country.jpg":
        "a real photograph of dry Texas hill country, golden brown grass field "
        "with large sprawling oak trees, distant purple hills, dramatic cumulus clouds"
        + PHOTO_SUFFIX,
    "robert-julian-onderdonk_sunlight-after-rain-1921.jpg":
        "a real photograph of warm sunlight filtering through large bare oak trees "
        "in an open park, wet muddy ground with puddles reflecting light, long shadows"
        + PHOTO_SUFFIX,
    "viktor-vasnetsov_ochtir-1879.jpg":
        "a real photograph of an old ornate stone balustrade terrace with weathered stairs "
        "in a garden, dense green trees and ivy in the background, warm golden hour light"
        + PHOTO_SUFFIX,
    "robert-julian-onderdonk_the-woodland-pool.jpg":
        "a real photograph of a calm woodland pond reflecting blue sky and clouds, "
        "tall pine and birch trees framing a sunlit clearing, lush green grass at the water edge"
        + PHOTO_SUFFIX,
}

GENERIC_PROMPT = (
    "a real DSLR photograph of a natural landscape scene, "
    "sharp focus, natural lighting, high dynamic range, 8k resolution, "
    "no painting, no illustration, no brushstrokes"
)


def load_and_preprocess(path, size=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    aspect = w / h
    if aspect > 1:
        new_w, new_h = size, max(round(size / aspect / 16) * 16, 16)
    else:
        new_w, new_h = max(round(size * aspect / 16) * 16, 16), size
    # keep square for memory simplicity
    new_w, new_h = size, size
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr)
    tensor = rearrange(tensor, "h w c -> 1 c h w")
    return tensor, new_w, new_h


def main():
    input_dir = os.environ.get("INPUT_DIR", "paintings")
    output_dir = os.environ.get("OUTPUT_DIR", "output_destylized")
    strength = float(os.environ.get("STRENGTH", "0.70"))
    num_steps = int(os.environ.get("NUM_STEPS", "35"))
    guidance = float(os.environ.get("GUIDANCE", "5.5"))
    seed = int(os.environ.get("SEED", "42"))
    size = int(os.environ.get("IMG_SIZE", "512"))
    name = "flux-dev"
    device = torch.device("cuda")

    os.makedirs(output_dir, exist_ok=True)

    images = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
    )
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} paintings to destylize")
    print(f"  strength={strength}, steps={num_steps}, guidance={guidance}, size={size}")

    # --- Phase 1: encode all text prompts ---
    print("\n=== Phase 1: Loading text encoders ===")
    t5 = LocalHFEmbedder(
        f"{LOCAL_MODELS}/t5-v1_1-xxl", max_length=512,
        is_clip=False, torch_dtype=torch.bfloat16,
    )
    clip = LocalHFEmbedder(
        f"{LOCAL_MODELS}/clip-vit-large-patch14", max_length=77,
        is_clip=True, torch_dtype=torch.bfloat16,
    )
    t5, clip = t5.to(device), clip.to(device)

    all_prompts = []
    for img_path in images:
        fname = os.path.basename(img_path)
        prompt = DEFAULT_PROMPTS.get(fname, GENERIC_PROMPT)
        all_prompts.append(prompt)
        print(f"  {fname} → {prompt[:80]}...")

    all_txt, all_vec = [], []
    with torch.inference_mode():
        for prompt in all_prompts:
            txt = t5([prompt])
            vec = clip([prompt])
            all_txt.append(txt.cpu())
            all_vec.append(vec.cpu())

    t5.cpu(); clip.cpu()
    del t5, clip
    gc.collect(); torch.cuda.empty_cache()
    print("Text encoders offloaded.")

    # --- Phase 2: encode all images with AE ---
    print("\n=== Phase 2: Encoding paintings with AE ===")
    ae = load_ae(name, device=device)

    all_z0 = []
    all_dims = []
    with torch.inference_mode():
        for img_path in images:
            img_tensor, w, h = load_and_preprocess(img_path, size=size)
            img_tensor = img_tensor.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z0 = ae.encode(img_tensor)
            all_z0.append(z0.cpu())
            all_dims.append((w, h))
            print(f"  Encoded {os.path.basename(img_path)}: z0 shape={z0.shape}")

    ae.cpu()
    del ae
    gc.collect(); torch.cuda.empty_cache()
    print("AE offloaded after encoding.")

    # --- Phase 3: denoise (flow model) ---
    print("\n=== Phase 3: Loading flow model & denoising ===")
    model = load_flow_model(name, device=device)
    print(f"GPU after flow model: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    all_denoised = []
    with torch.inference_mode():
        for i, (img_path, z0, (w, h), txt, vec) in enumerate(
            zip(images, all_z0, all_dims, all_txt, all_vec)
        ):
            fname = os.path.basename(img_path)
            print(f"\n  [{i+1}/{len(images)}] Denoising {fname}...")

            z0 = z0.to(device)
            noise = get_noise(1, h, w, device=device, dtype=torch.bfloat16, seed=seed + i)

            # img2img: mix encoded image with noise at t=strength
            t_start = strength
            z_noisy = (1 - t_start) * z0 + t_start * noise

            img_packed = rearrange(z_noisy, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            bs = 1
            hh = z_noisy.shape[2]
            ww = z_noisy.shape[3]
            img_ids = torch.zeros(hh // 2, ww // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(hh // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(ww // 2)[None, :]
            img_ids = img_ids.unsqueeze(0).to(device)
            img_ids = rearrange(img_ids, "b h w c -> b (h w) c")

            txt_gpu = txt.to(device)
            vec_gpu = vec.to(device)
            txt_ids = torch.zeros(bs, txt_gpu.shape[1], 3).to(device)

            image_seq_len = img_packed.shape[1]
            full_timesteps = get_schedule(num_steps, image_seq_len, shift=True)

            # truncate schedule to start from ~strength
            init_steps = max(int(num_steps * strength), 1)
            start_idx = num_steps - init_steps
            timesteps = full_timesteps[start_idx:]

            x = denoise(
                model,
                img=img_packed,
                img_ids=img_ids,
                txt=txt_gpu,
                txt_ids=txt_ids,
                vec=vec_gpu,
                timesteps=timesteps,
                guidance=guidance,
            )
            all_denoised.append((x.cpu(), w, h, fname))

            del z0, noise, z_noisy, img_packed, img_ids, txt_gpu, vec_gpu, txt_ids, x
            torch.cuda.empty_cache()

    model.cpu()
    del model
    gc.collect(); torch.cuda.empty_cache()
    print("\nFlow model offloaded.")

    # --- Phase 4: decode with AE ---
    print("\n=== Phase 4: Decoding with AE ===")
    ae = load_ae(name, device=device)

    with torch.inference_mode():
        for x_cpu, w, h, fname in all_denoised:
            x = x_cpu.to(device)
            x = unpack(x.float(), h, w)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = ae.decode(x)
            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

            out_name = f"destylized_{os.path.splitext(fname)[0]}.jpg"
            out_path = os.path.join(output_dir, out_name)
            img.save(out_path, quality=95, subsampling=0)
            print(f"  Saved {out_path}")

            del x, img
            torch.cuda.empty_cache()

    print(f"\nDone! {len(all_denoised)} images saved to {output_dir}/")


if __name__ == "__main__":
    main()
