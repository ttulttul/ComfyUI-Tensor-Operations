import logging
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
    
NODE_CLASS_MAPPINGS = {
    "Image Match Normalize": ImageMatchNormalize,
    "Latent Match Normalize": LatentMatchNormalize
}