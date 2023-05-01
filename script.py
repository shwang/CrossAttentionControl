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


# TODO(shwang): nonglobal variables please (loading takes forever)
# Ideally, we can import into Jupyter notebook from this script.

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


def init_attention_weights(weight_tuples):
    ## By default, called with weight_tuples=[], leading to all weights being 1.0.
    tokens_length = clip_tokenizer.model_max_length
    weights = torch.ones(tokens_length)
    
    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None
    

def init_attention_edit(tokens, tokens_edit):
    tokens_length = clip_tokenizer.model_max_length
    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)

    tokens = tokens.input_ids.numpy()[0]
    tokens_edit = tokens_edit.input_ids.numpy()[0]
    
    for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_mask = mask.to(device)
            module.last_attn_slice_indices = indices.to(device)    # Length T. All zeros, except replacement tokens.
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_mask = None
            module.last_attn_slice_indices = None


def init_attention_func():
    def new_attention(self, query, key, value):
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale  ## shape: (batch*n_heads, seq_len, seq_len)
        ## query shape: (B*H, n_pixels, d_head)
        ## key.T shape: (B*H, d_head, T)    ==> multiplies to (B*H, T, T)
        attn_slice = attention_scores.softmax(dim=-1)    ## shape: (B*H, T, T). Final dim adds up to 1.0 for every (i, j).

        # Proposition: self.to_k(left_right) instead.

        # compute attention output
        n_pixels = query.shape[1]
        W = H = int(n_pixels ** 0.5)
        assert W**2 == n_pixels

        mask_left = torch.zeros(H, W, dtype=bool)
        mask_left[:, 0:int(W//2)] = 1
        mask_left = mask_left.reshape(1, n_pixels, 1)
        mask_left = mask_left.to(device)
        mask_right = ~mask_left

        def proc_attention(attn: torch.Tensor):
            return attn.detach().mean(dim=0).cpu().numpy()

        if self.save_left_attn:
            assert not self.save_right_attn
            self.left_attentions_saved.append(proc_attention(attn_slice))
        elif self.save_right_attn:
            self.right_attentions_saved.append(proc_attention(attn_slice))

        SPECIAL_HIDDEN_STATES = False
        
        ## Begin editted code.
        if self.use_last_attn_slice:
            if self.last_attn_slice_mask is not None:   ## NOTE: last_attn === "edit prompt"
                new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                # print(query.shape, key.shape, value.shape)
                # print(attn_slice.shape)  # BH, n_pixels, T
                if not OUR_EXPERIMENT:
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = attn_slice * (mask_right) + self.last_attn_slice * mask_left

                left_scores = torch.matmul(query, self.left_key.transpose(-1, -2)) * self.scale  ## shape: (batch*n_heads, seq_len, seq_len)
                mask_left = einops.repeat(mask_left, "1 P 1 -> B P T", B=left_scores.shape[0], T=left_scores.shape[2])
                mask_right = einops.repeat(mask_right, "1 P 1 -> B P T", B=left_scores.shape[0], T=left_scores.shape[2])
                left_scores[mask_right] = float('-inf')
                right_scores = attention_scores
                right_scores[mask_left] = float('-inf')

                left_attn_probs = left_scores.softmax(dim=-1)    ## shape: (B*H, T, T). Final dim adds up to 1.0 for every (i, j).
                right_attn_probs = right_scores.softmax(dim=-1)    ## shape: (B*H, T, T). Final dim adds up to 1.0 for every (i, j).
            else:  # mask is None ==> attn1 module, or self-attention between pixels.
                attn_slice = self.last_attn_slice

            self.final_attentions_saved.append(proc_attention(attn_slice))

            SPECIAL_HIDDEN_STATES = True

            self.use_last_attn_slice = False

        if self.save_last_attn_slice:  ## Hacky!!!
            self.last_attn_slice = attn_slice
            self.save_last_attn_slice = False
            self.left_key = key
            self.left_value = value

        if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
            attn_slice = attn_slice * self.last_attn_slice_weights
            self.use_last_attn_weights = False
        
        ## Resumes original code.
        hidden_states = torch.matmul(attn_slice, value)   # Reshape attn_slice (BH, 1, T)  => (BH, T, 1)

        if SPECIAL_HIDDEN_STATES:
            left_side = torch.matmul(left_attn_probs, self.left_value).nan_to_num(nan=0.0)
            right_side = torch.matmul(right_attn_probs, value).nan_to_num(nan=0.0)
            hidden_states = left_side + right_side
            # assert not torch.isnan(hidden_states).any().item()

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # [8, 4096, 40]
        return hidden_states
    
    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        # We don't implement this case (optimized case) but want to error out when it is called.
        raise NotImplementedError

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.left_prompt = None
            module.right_prompt = None
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))
            module.final_attentions_saved = []
            module.left_attentions_saved = []
            module.right_attentions_saved = []
            module.save_left_attn = module.save_right_attn = False

def use_last_tokens_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_slice = use
            
def use_last_tokens_attention_weights(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use
            
def use_last_self_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.use_last_attn_slice = use
            
def save_last_tokens_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.save_last_attn_slice = save
            
def save_last_self_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.save_last_attn_slice = save

def viz_save_left_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.save_left_attn = save

def viz_save_right_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.save_right_attn = save

def get_saved_attention_maps() -> Dict[str, Dict[str, np.ndarray]]:
    """
    {"left", "right", "final"}
    => {up_blocks.3.attnetions.1.transformer_blocks.0.attn1}
    => np.ndarray of shape [n_pixels, n_prompt_tokens] IE [W**2, T].
        (later, also [W**2, W**2]).
    """

    left = {}
    right = {}
    final = {}
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            left[name] = module.left_attentions_saved
            right[name] = module.right_attentions_saved
            final[name] = module.final_attentions_saved
    result = dict(left=left, right=right, final=final)
    return result
            
@torch.no_grad()
def stablediffusion(prompt="", prompt_edit=None, prompt_edit_token_weights=[], prompt_edit_tokens_start=0.0, prompt_edit_tokens_end=1.0, prompt_edit_spatial_start=0.0, prompt_edit_spatial_end=1.0, guidance_scale=7.5, steps=50, seed=None, width=512, height=512, init_image=None, init_image_strength=0.5,
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
        tokens_unconditional = clip_tokenizer("", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

        #Process prompt editing
        if prompt_edit is not None:
            tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state
            
            ## Would replace with `UNet.init_attention_edit`.
            ## Would also wrap the UNet, perhaps.
            init_attention_edit(tokens_conditional, tokens_conditional_edit)
            
        init_attention_func()
        init_attention_weights(prompt_edit_token_weights)
            
        timesteps = scheduler.timesteps[t_start:]
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            #sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            #Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
            #Prepare the Cross-Attention layers
            if prompt_edit is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                #Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()
                
            #Predict the conditional noise residual and save the cross-attention layer activations
            viz_save_left_attention(save_attentions and True)
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            viz_save_left_attention(False)


            left_prompt = embedding_conditional
            assert prompt_edit is not None
            _tokens_conditional = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            _embedding_conditional = clip(_tokens_conditional.input_ids.to(device)).last_hidden_state
            right_prompt = _embedding_conditional

            def set_left_right_prompt_embs(left, right):
                assert isinstance(left, torch.Tensor)
                for name, module in unet.named_modules():
                    module.left_emb = left
                    module.right_emb = right

            set_left_right_prompt_embs(left_prompt, right_prompt)
            
            #Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / scheduler.num_train_timesteps
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()
                    
                #Use weights on edited prompt
                use_last_tokens_attention_weights()

                #Predict the edited conditional noise residual using the cross-attention masks
                # viz_save_right_attention(save_attentions and True)

                # encoder_hidden_states here will be unused. We will use instance variablef left and right instead.

                noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample
                # viz_save_right_attention(False)
                
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


def make_unique_filename() -> str:
    import datetime
    import randomname
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    rand_name = randomname.get_name()
    return f"{timestamp}_{rand_name}"


def main_inspect_attentions():
    stablediffusion("a cat sitting on a car", "a smiling dog sitting on a car", seed=248396402679,
                    prompt_edit_spatial_start=999, save_attentions=True)
    maps = get_saved_attention_maps()

    import numpy
    key = "down_blocks.0.attentions.0.transformer_blocks.0.attn2"
    tup = left, right, final = maps["left"][key], maps["right"][key], maps["final"][key]
    path = "attention.npy"
    numpy.save("attention.npy", dict(left=left, right=right, final=final))
    print(f"Saved down0 attentions to {path}")

    left = maps["left"]
    # Print the shape of every value in left alongside the key
    def inspect(x):
        for k, v in x.items():
            print(k, np.array(v).shape)

    breakpoint()
    for k in "left right final".split():
        print()
        print(k)
        inspect(maps[k])

def main():
    stablediffusion("a cat sitting on a car", "a smiling dog sitting on a car", seed=248396402679,
                    prompt_edit_spatial_start=999, save_attentions=False)
    

if __name__ == "__main__":
    main()