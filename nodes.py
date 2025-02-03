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

    # Quantize the image to reduce colors
    image = image.quantize(colors=num_colors)
    image = image.convert("RGBA")

    # Convert image to tensor
    pixel_data = np.array(image)
    tensor_image = torch.from_numpy(pixel_data).float().cuda()

    print(f'tensor_image: {tensor_image.shape}')

    # If no mask provided, use the entire image for the color palette
    if mask_tensor is None:
        mask_tensor = torch.zeros((image.height, image.width), dtype=torch.float32, device='cuda')
        print(f'image2noise: created empty mask tensor: {mask_tensor.shape}')
    else:
        # Ensure mask is on CPU for resize operation
        mask_tensor = mask_tensor.cpu()

        # Check mask dimensions
        if len(mask_tensor.shape) != 2:
            raise ValueError(f"Expected mask_tensor to have 2 dimensions (H,W), got shape {mask_tensor.shape}")
            
        # Check if resize is needed
        if mask_tensor.shape != (image.height, image.width):
            print(f"Resizing mask from {mask_tensor.shape} to ({image.height}, {image.width})")
            mask_tensor = resize_mask(mask_tensor, image.height, image.width)

        # Move to CUDA after processing
        mask_tensor = mask_tensor.cuda()
        print(f'image2noise: resized mask tensor to: {mask_tensor.shape}')

    # Create a flattened index of unmasked pixels
    flat_mask = mask_tensor.reshape(-1)
    unmasked_indices = torch.nonzero(flat_mask == 0).squeeze(1)

    print(f'image2noise: unmasked_indices has length {len(unmasked_indices)}')

    if len(unmasked_indices) > 0:
        # Reshape tensor_image to [H*W, 4]
        flat_image = tensor_image.reshape(-1, 4)

        # Get unmasked pixels
        unmasked_pixels = flat_image[unmasked_indices]

        # Shuffle unmasked pixels
        if len(unmasked_indices) > 1:
            perm = torch.randperm(len(unmasked_indices), device='cuda')
            shuffled_pixels = unmasked_pixels[perm]

            # Place shuffled pixels back in their unmasked positions
            flat_image[unmasked_indices] = shuffled_pixels
        else:
            print(f'image2noise: not enough masked pixels to shuffle')

        # Reshape back to original shape
        tensor_image = flat_image.reshape(tensor_image.shape)
    else:
        print(f'image2noise: unmasked_indices is empty')

    # Create black noise tensor only in unmasked regions
    if black_mix > 0.0:
        random_tensor = torch.randn_like(tensor_image[..., :3])
        black_mask = (random_tensor < black_mix) & (mask_tensor.unsqueeze(-1) == 0)
        tensor_image[..., :3][black_mask] = 0

    # Apply brightness enhancement only to unmasked regions
    brightness_mask = (mask_tensor == 0).unsqueeze(-1).expand(-1, -1, 3)
    tensor_image[..., :3][brightness_mask] *= brightness

    # Apply Gaussian blur if specified, respecting the mask
    if gaussian_mix > 0:
        import torch.nn.functional as F
        kernel_size = int(gaussian_mix * 2 + 1)
        padding = kernel_size // 2
        gaussian_kernel = torch.exp(-0.5 * (torch.arange(-padding, padding + 1, dtype=torch.float32, device='cuda') ** 2) / gaussian_mix ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, -1)

        # Create a mask for the blur operation
        blur_mask = (mask_tensor == 0).float()

        for i in range(3):
            channel = tensor_image[..., i].clone()

            # Apply mask before blurring
            channel = channel * blur_mask

            # Add batch and channel dimensions for conv2d
            channel = channel.unsqueeze(0).unsqueeze(0)

            # Apply blur
            blurred = F.pad(channel, (padding, padding, padding, padding), mode='reflect')
            blurred = F.conv2d(blurred, gaussian_kernel.view(1, 1, -1, 1))
            blurred = F.conv2d(blurred, gaussian_kernel.view(1, 1, 1, -1))

            # Remove batch and channel dimensions
            blurred = blurred.squeeze(0).squeeze(0)

            # Only update unmasked regions
            unmasked = blur_mask == 1
            tensor_image[..., i][unmasked] = blurred[unmasked]

    # Convert tensor back to image
    tensor_image = tensor_image.clamp(0, 255).byte().cpu().numpy()
    randomized_image = Image.fromarray(tensor_image)

    return randomized_image

def image2noise_new(
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
    if num_channels != 4:
        raise ValueError(f"Expected tensor_image to have 4 channels (RGBA), got {num_channels}")

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

    # Reshape tensor_image to [H*W, C]
    flat_image = tensor_image.reshape(-1, num_channels)
    flat_mask = mask_tensor.reshape(-1)

    # Extract colors from unmasked regions for palette generation
    unmasked_indices = torch.nonzero(flat_mask == 0).squeeze(1)

    # Get color palette from unmasked regions
    if len(unmasked_indices) > 0:
        # Ensure we don't try to select more colors than we have pixels
        num_available_colors = len(unmasked_indices)
        actual_num_colors = min(num_colors, num_available_colors)

        # Randomly select pixels from unmasked regions for the palette
        if num_available_colors > actual_num_colors:
            palette_indices = torch.randperm(num_available_colors, device='cuda')[:actual_num_colors]
            selected_indices = unmasked_indices[palette_indices]
        else:
            selected_indices = unmasked_indices

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
