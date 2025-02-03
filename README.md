# ComfyUI Tensor Operations Nodes

This repo contains nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implement some helpful operations on tensors, such as normalization.

## Updates

### June 12th, 2024

- Added `Fast Image to Noise` node, which generates a new image that is effectively a remix of the pixels in a source image.
- This is all done on the GPU and is lightning fast in comparison with the [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) `Image to Noise` node.

### February 2nd, 2024

- Initial release. Added `Image Match Normalize` and `Latent Match Normalize` nodes.

## Nodes

### Image Match Normalize

This node returns a normalized version of the target image using the mean and standard deviation of each color channel of the source image. If you want the color and brightness of your image to match the colour and brightness of another image,
this is the node for you.

### Latent Match Normalize

This node returns a normalized version of the target latent using the mean and standard deviation of each channel of
the source latent. Latents encode color information differently than images and you may find that normalizing an image
by instead normalizing its latent representation results in a "better" result that is closer to the coloring of the
source.

### Fast Image to Noise

Generates a new image by remixing the pixels of the source image randomly,
with several customization options:

- **num_colors**: Number of colors to sample for the noise palette (2-256)
- **black_mix**: Probability of replacing pixels with black (0-1)
- **gaussian_mix**: Amount of Gaussian blur to apply (0-1024)
- **brightness**: Brightness adjustment factor (0-2)
- **mask**: Optional mask input that restricts color palette selection to masked areas only
- **output_mode**: Choose between "batch" or "list" output format
- **seed**: Random seed for reproducible results

The node is intended to match the functionality of the `Image to Noise` node in the [WAS
Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui) but operates entirely
on the GPU for significantly faster processing. When a mask is provided, the node will only
sample colors from the masked (white) areas of the image to create the noise palette,
allowing for more targeted noise generation.
