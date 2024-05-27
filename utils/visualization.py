
import torch
from torchvision import transforms, utils

def plot_tensor_image_torch(tensor_image, title=None):
    """
    Plots a PyTorch tensor image using only PyTorch functions.

    Args:
        tensor_image (torch.Tensor): The image tensor to plot.
        title (str, optional): Title for the plot (not displayed directly, only for reference). Defaults to None.
    """

    # Check if tensor is on GPU, move to CPU if necessary
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()

    # Convert tensor to PIL Image (this will handle the denormalization implicitly)
    pil_image = transforms.ToPILImage()(tensor_image)

    # Display image (this will open the image in the default image viewer)
    # pil_image.show(title=title)
    return pil_image