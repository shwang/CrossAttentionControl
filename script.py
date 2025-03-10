from typing import Dict
import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
import numpy as np
import random
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher
import einops

"""
Overwrites CrossAttention._sliced_attention and CrossAttention._attention.
"""
def init_attention_func(unet, vae, clip, device, clip_tokenizer):
    def new_attention(self: "CrossAttention", query, key, value):
        del key, value  # We ignore these arguments! Instead, we accept them as hardcoded instance variables in `self.mapping`.
        ## query shape: (B*H, n_pixels, dim_head)
        ## key.T shape: (B*H, dim_head, T)

        # TODO: Check whether `forward` uses context. This would explain why changing the hidden_state argument unexpectedly
        #       causes changes.

        n_pixels = query.shape[1]
        W = H = int(n_pixels ** 0.5)
        assert W**2 == n_pixels
        hidden_states = 0.

        # We create self.mappings as a List[Tuple[torch.Tensor, Callable[[int, int], torch.Tensor]]]
        for prompt_emb, mask_fn in self.mappings:
            mask = mask_fn(W, H)
            mask = mask.reshape(1, n_pixels, 1)
            mask.to(device)

            k = self.reshape_heads_to_batch_dim(self.to_k(prompt_emb))  # Project the prompt embedding to the key space (including multiple heads).
            v = self.reshape_heads_to_batch_dim(self.to_v(prompt_emb))  # Project the prompt embedding to the key space (including multiple heads).
            scores = torch.matmul(query, k.transpose(-1, -2)) * self.scale  ## shape: (batch*n_heads, n_pixels, seq_len)
            mask = einops.repeat(mask, "1 P 1 -> B P T", B=scores.shape[0], T=scores.shape[2])  # Expand indices for indexing two lines down.
            scores[~mask] = -float("inf")
            attn_probs = scores.softmax(dim=-1)
            attn_probs[~mask] = 0  # Replace nans with 0s.
            hidden_states += torch.matmul(attn_probs, v)   # Reshape attn_slice (BH, 1, T)  => (BH, T, 1)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # [8, 4096, 40]
        return hidden_states
    
    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        # We don't implement this case (slicing optimized case) but want to error out when it is called.
        raise NotImplementedError

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:   # "attn2" is cross-attention. "attn1" is (equivalent to) self-attention.
            module.mapping = None
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))


@torch.no_grad()
def stablediffusion(left_prompt, right_prompt, *, unet, vae, clip, device, clip_tokenizer,
                    guidance_scale=7.5, steps=50, seed=None, width=512, height=512, init_image=None, init_image_strength=0.5,
        save_attentions=False):
    #Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    #Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    
    #Preprocess image if it exists (img2img)
    if init_image is not None:
        #Resize and transpose for numpy b h w c -> torch b c h w
        init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))
        
        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])
            
        #Move image to GPU
        init_image = init_image.to(device)
        
        #Encode image
        with autocast(device):
            init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
            
        t_start = steps - int(steps * init_image_strength)   # Nice, we start from image latent.
    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_start = 0
    
    #Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    #latent = noise * scheduler.init_noise_sigma
    latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=device)).to(device)
    
    #Process clip
    with autocast(device):
        def get_prompt_emb(prompt):
            tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state
            return embedding_conditional

        uncond_emb = get_prompt_emb("")
        left_emb = get_prompt_emb(left_prompt)
        right_emb = get_prompt_emb(right_prompt)
        dummy_emb = torch.zeros_like(uncond_emb).to(device)

        init_attention_func(unet=unet, clip=clip, vae=vae, device=device, clip_tokenizer=clip_tokenizer)

        # Assign the paired prompt embeddings and mask-making functions to each cross-attention layer.
        # The new_attention() function will read these embeddings and masks and ignore its encoder_hidden_state argument.
        def use_unconditional_mappings():
            # Mapping associated with unconditional denoising.
            for name, module in unet.named_modules():
                if type(module).__name__ == "CrossAttention" and "attn2" in name:
                    module.mappings = ((uncond_emb, make_true_mask),)
                else:
                    module.mappings = None

        def use_conditional_mappings():
            # Mapping associated with conditional denoising (includes separate left and right embeddings).
            for name, module in unet.named_modules():
                if type(module).__name__ == "CrossAttention" and "attn2" in name:
                    module.mappings = ((left_emb, make_left_mask), (right_emb, make_right_mask))
                else:
                    module.mappings = None
            
        timesteps = scheduler.timesteps[t_start:]
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            #sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            #Predict the unconditional noise residual

            use_unconditional_mappings()
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=dummy_emb).sample
            
            #Prepare the Cross-Attention layers
            #Predict the conditional noise residual and save the cross-attention layer activations
            use_conditional_mappings()
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=dummy_emb).sample
            # dummy_emb is not (should not be) used in calculations since encoder_hidden_state is supposed to be ignored.
            # TODO: sanity check -- result should be the same if we use encoder_hidden_states=left_emb too.

            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def make_left_mask(W: int, H: int) -> torch.Tensor:
    mask_left = torch.zeros(H, W, dtype=bool)
    mask_left[:, 0:int(W//2)] = 1
    return mask_left


def make_right_mask(W: int, H:int) -> torch.Tensor:
    mask_right = torch.zeros(H, W, dtype=bool)
    mask_right[:, int(W//2):] = 1
    return mask_right


def make_true_mask(W: int, H:int) -> torch.Tensor:
    return torch.ones(H, W, dtype=bool)


def load_models():
    #Init CLIP tokenizer and model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
    clip = clip_model.text_model

    #Init diffusion model
    # Using Steven's huggingface auth token.
    auth_token = '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' #Replace this with huggingface auth token as a string if model is not already downloaded
    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)

    #Move to GPU
    device = "cuda"
    unet.to(device)
    vae.to(device)
    clip.to(device)
    print("Loaded all models")

    return unet, vae, clip, clip_tokenizer, device


def main():
    unet, vae, clip, clip_tokenizer, device = load_models()
    stablediffusion("a cat", "a dog", unet=unet, vae=vae, device=device, clip=clip,
                    clip_tokenizer=clip_tokenizer,
                    seed=248396402679)
    

if __name__ == "__main__":
    main()