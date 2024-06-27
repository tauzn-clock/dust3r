import torch
import numpy as np
import PIL.Image as Image

def load_masks(masks_array, H, W, device):
    masks = []
    for i in range(len(masks_array)):
        mask = Image.open(masks_array[i]).convert('L')
        mask = mask.resize((W,H))

        mask = np.array(mask)
        mask = torch.tensor(mask).to(device)/255
        masks.append(mask)

    return masks