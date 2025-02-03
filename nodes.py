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

def resize_mask(mask_tensor, target_height, target_width):
    """Resize mask tensor to target dimensions using PIL"""
    # Convert tensor to PIL Image
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np)

    # Resize using nearest neighbor to preserve binary values
    resized_mask = mask_pil.resize((target_width, target_height), Image.Resampling.NEAREST)

    # Convert back to tensor
    resized_np = np.array(resized_mask).astype(np.float32) / 255.0
    return torch.from_numpy(resized_np)


def image2noise(
    image: Image.Image,
    mask_tensor: torch.Tensor = None,
    num_colors: int = 16,
    black_mix: float = 0.0,
    brightness: float = 1.0,
    gaussian_mix: float = 0.0,
    seed: int = 0
) -> Image.Image:
    # Set the seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Convert image to tensor
    pixel_data = np.array(image)
    tensor_image = torch.from_numpy(pixel_data).float().cuda()

    # Store original shape for later reshaping
    original_shape = tensor_image.shape
    if len(original_shape) != 3:
        raise ValueError(f"Expected tensor_image to have 3 dimensions (H,W,C), got shape {original_shape}")
    
    num_channels = original_shape[-1]
    if num_channels != 3:
        raise ValueError(f"Expected tensor_image to have 3 channels (RGB), got {num_channels}")

    # If no mask provided, use the entire image for color palette
    if mask_tensor is None:
        mask_tensor = torch.zeros((image.height, image.width), dtype=torch.float32, device='cuda')
    else:
        # Ensure mask is on CPU for resize operation
        mask_tensor = mask_tensor.cpu()

        # Check if resize is needed
        if mask_tensor.shape != (image.height, image.width):
            mask_tensor = resize_mask(mask_tensor, image.height, image.width)

        # Move to CUDA after processing
        mask_tensor = mask_tensor.cuda()

    # Validate mask tensor shape
    if mask_tensor is not None:
        if len(mask_tensor.shape) != 2:
            raise ValueError(f"Expected mask_tensor to have 2 dimensions (H,W), got shape {mask_tensor.shape}")
        if mask_tensor.shape != (image.height, image.width):
            raise ValueError(f"Mask shape mismatch. Expected ({image.height}, {image.width}), got {mask_tensor.shape}")

    # Reshape tensor_image to [H*W, C]
    flat_image = tensor_image.reshape(-1, num_channels)
    flat_mask = mask_tensor.reshape(-1)
    
    # Validate flattened shapes
    expected_pixels = original_shape[0] * original_shape[1]
    if flat_image.shape[0] != expected_pixels:
        raise ValueError(f"Flattened image has wrong number of pixels. Expected {expected_pixels}, got {flat_image.shape[0]}")
    if flat_mask.shape[0] != expected_pixels:
        raise ValueError(f"Flattened mask has wrong number of pixels. Expected {expected_pixels}, got {flat_mask.shape[0]}")

    # Extract colors from masked regions for palette generation
    masked_indices = torch.nonzero(flat_mask == 1).squeeze(1)
    
    if masked_indices.dim() != 1:
        raise ValueError(f"Unexpected masked_indices shape. Expected 1D tensor, got {masked_indices.dim()}D")

    # Get color palette from masked regions
    if len(masked_indices) > 0:
        # Ensure we don't try to select more colors than we have pixels
        num_available_colors = len(masked_indices)
        actual_num_colors = min(num_colors, num_available_colors)

        # Randomly select pixels from masked regions for the palette
        if num_available_colors > actual_num_colors:
            palette_indices = torch.randperm(num_available_colors, device='cuda')[:actual_num_colors]
            selected_indices = masked_indices[palette_indices]
        else:
            selected_indices = masked_indices

        color_palette = flat_image[selected_indices]
    else:
        # If mask covers everything, sample from the entire image
        num_available_colors = len(flat_image)
        actual_num_colors = min(num_colors, num_available_colors)
        palette_indices = torch.randperm(num_available_colors, device='cuda')[:actual_num_colors]
        color_palette = flat_image[palette_indices]

    # Generate random indices for the entire image
    num_pixels = original_shape[0] * original_shape[1]
    random_indices = torch.randint(0, len(color_palette), (num_pixels,), device='cuda')

    # Validate color palette shape
    if len(color_palette.shape) != 2 or color_palette.shape[1] != num_channels:
        raise ValueError(f"Color palette has invalid shape. Expected (N,{num_channels}), got {color_palette.shape}")
        
    # Create new noise image using the color palette
    noise_image = color_palette[random_indices].reshape(original_shape)
    
    if noise_image.shape != original_shape:
        raise ValueError(f"Shape mismatch after palette lookup. Expected {original_shape}, got {noise_image.shape}")

    # Apply black mix if specified
    if black_mix > 0.0:
        black_mask = torch.rand_like(noise_image[..., 0]) < black_mix
        noise_image[black_mask, :3] = 0

    # Apply brightness adjustment
    noise_image[..., :3] *= brightness

    # Apply Gaussian blur if specified
    if gaussian_mix > 0:
        if len(noise_image.shape) != 3:
            raise ValueError(f"Expected noise_image to have 3 dimensions before Gaussian blur, got shape {noise_image.shape}")
        import torch.nn.functional as F
        kernel_size = int(gaussian_mix * 2 + 1)
        padding = kernel_size // 2
        gaussian_kernel = torch.exp(-0.5 * (torch.arange(-padding, padding + 1, dtype=torch.float32, device='cuda') ** 2) / gaussian_mix ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, -1)

        for i in range(3):  # Only blur RGB channels, not alpha
            channel = noise_image[..., i].clone()
            channel = channel.unsqueeze(0).unsqueeze(0)
            blurred = F.pad(channel, (padding, padding, padding, padding), mode='reflect')
            blurred = F.conv2d(blurred, gaussian_kernel.view(1, 1, -1, 1))
            blurred = F.conv2d(blurred, gaussian_kernel.view(1, 1, 1, -1))
            noise_image[..., i] = blurred.squeeze(0).squeeze(0)

    # Convert tensor back to image
    noise_image = noise_image.clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(noise_image)

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
            "optional": {
                "mask": ("MASK",),  # Made mask optional
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "image_to_noise"

    CATEGORY = "WAS Suite/Image/Generate/Noise"

    def image_to_noise(self, images, num_colors, black_mix, gaussian_mix, brightness, output_mode, seed, mask=None):
        noise_images = []
        for i, image in enumerate(images):
            # Get corresponding mask for this image if mask is provided
            current_mask = mask[i] if mask is not None and len(mask) > i else None
            noise_images.append(pil2tensor(image2noise_new(tensor2pil(image), current_mask, num_colors, black_mix, brightness, gaussian_mix, seed)))
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
