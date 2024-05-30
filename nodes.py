import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import torch

@torch.no_grad()
def match_normalize(target_tensor, source_tensor, dimensions=4):
    "Adjust target_tensor based on source_tensor's mean and stddev"   
    if len(target_tensor.shape) != dimensions:
        raise ValueError("source_latent must have four dimensions")
    if len(source_tensor.shape) != dimensions:
        raise ValueError("target_latent must have four dimensions")

    # Put everything on the same device
    device = target_tensor.device

    # Calculate the mean and std of target tensor
    tgt_mean = target_tensor.mean(dim=[2, 3], keepdim=True).to(device)
    tgt_std = target_tensor.std(dim=[2, 3], keepdim=True).to(device)
    
    # Calculate the mean and std of source tensor
    src_mean = source_tensor.mean(dim=[2, 3], keepdim=True).to(device)
    src_std = source_tensor.std(dim=[2, 3], keepdim=True).to(device)
    
    # Normalize target tensor to have mean=0 and std=1, then rescale
    normalized_tensor = (target_tensor.clone() - tgt_mean) / tgt_std * src_std + src_mean
    
    return normalized_tensor

class LatentMatchNormalize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"source_latent": ("LATENT", ),
                     "target_latent": ("LATENT", )}}
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "tensor_ops"

    FUNCTION = "latent_match_normalize"

    @torch.no_grad()
    def latent_match_normalize(self, source_latent, target_latent):       
        normalized_latent = match_normalize(target_latent["samples"], source_latent["samples"], dimensions=4)

        return_latent = source_latent.copy()
        return_latent["samples"] = normalized_latent
        return (return_latent,)
    
class ImageMatchNormalize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"source_image": ("IMAGE", ),
                     "target_image": ("IMAGE", )}}
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "tensor_ops"

    FUNCTION = "image_match_normalize"

    @torch.no_grad()
    def image_match_normalize(self, source_image, target_image):
        # image shape is [B, H, W, C], but the normalize function needs [B, C, H, W]
        source = source_image.permute(0,3,1,2)
        target = target_image.permute(0,3,1,2)
        
        normalized = match_normalize(target, source, dimensions=4)

        normalized_image = normalized.permute(0,2,3,1)
        return (normalized_image,)

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image2noise(
    image: Image.Image,
    num_colors: int = 16,
    black_mix: float = 0.0,
    brightness: float = 1.0,
    gaussian_mix: float = 0.0,
    seed: int = 0
) -> Image.Image:
    # Set the seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Quantize the image to reduce colors
    image = image.quantize(colors=num_colors)
    image = image.convert("RGBA")

    # Convert image to tensor
    pixel_data = np.array(image)
    tensor_image = torch.from_numpy(pixel_data).float().cuda()

    # Randomly shuffle pixels
    perm = torch.randperm(tensor_image.nelement() // 4).cuda()
    tensor_image = tensor_image.view(-1, 4)[perm].view(*tensor_image.shape)

    # Create black noise tensor
    if black_mix > 0.0:
        # Ignore the alpha channel.
        random_tensor = torch.randn_like(tensor_image[:3, :, :])
        mask = random_tensor < black_mix
        tensor_image[:3, :, :][mask] = 0

    # Apply brightness enhancement
    tensor_image[:, :, :3] = tensor_image[:, :, :3] * brightness

    # Apply Gaussian blur if specified
    if gaussian_mix > 0:
        import torch.nn.functional as F
        kernel_size = int(gaussian_mix * 2 + 1)
        padding = kernel_size // 2
        gaussian_kernel = torch.exp(-0.5 * (torch.arange(-padding, padding + 1, dtype=torch.float32) ** 2) / gaussian_mix ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, -1).cuda()

        for i in range(3):
            channel = tensor_image[:, :, i].unsqueeze(0).unsqueeze(0)
            blurred = F.pad(channel, (padding, padding, padding, padding), mode='reflect')
            blurred = F.conv2d(blurred, gaussian_kernel.view(1, 1, -1, 1), padding=0, stride=1)
            blurred = F.conv2d(blurred, gaussian_kernel.view(1, 1, 1, -1), padding=0, stride=1)
            tensor_image[:, :, i] = blurred.squeeze(0).squeeze(0)[:, :tensor_image.shape[1]]

    # Convert tensor back to image
    tensor_image = tensor_image.clamp(0, 255).byte().cpu().numpy()
    randomized_image = Image.fromarray(tensor_image)

    return randomized_image


class ImageToNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_colors": ("INT", {"default": 16, "max": 256, "min": 2, "step": 2}),
                "black_mix": ("FLOAT", {"default": 0.0, "max": 1.0, "min": 0.0, "step": 0.1}),
                "gaussian_mix": ("FLOAT", {"default": 0.0, "max": 1024, "min": 0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.0, "step": 0.01}),
                "output_mode": (["batch","list"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "image_to_noise"

    CATEGORY = "WAS Suite/Image/Generate/Noise"

    def image_to_noise(self, images, num_colors, black_mix, gaussian_mix, brightness, output_mode, seed):
        noise_images = []
        for image in images:
            noise_images.append(pil2tensor(image2noise(tensor2pil(image), num_colors, black_mix, brightness, gaussian_mix, seed)))
        if output_mode == "list":
            self.OUTPUT_IS_LIST = (True,)
        else:
            noise_images = torch.cat(noise_images, dim=0)
        return (noise_images, )

NODE_CLASS_MAPPINGS = {
    "Image Match Normalize": ImageMatchNormalize,
    "Latent Match Normalize": LatentMatchNormalize,
    "Fast Image to Noise": ImageToNoise,
}
