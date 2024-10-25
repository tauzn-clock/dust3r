import torch
import torch.nn as nn
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.inference import *

@torch.no_grad()
def inference_weighted(pairs, model, device, cut_off = 1, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), model, None, device)

        pts3d_1 = res["pred1"]["pts3d"][0]
        pts3d_2 = res["pred2"]["pts3d_in_other_view"][0]

        weight_1 = pts3d_to_weight(pts3d_1, cut_off)
        weight_2 = pts3d_to_weight(pts3d_2, cut_off)
        
        res["pred1"]["conf"] = weighted_conf(res["pred1"]["conf"],weight_1)
        res["pred2"]["conf"] = weighted_conf(res["pred2"]["conf"],weight_2)

        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result

def pts3d_to_weight(pts3d, cut_off=1):
    pts3d = torch.linalg.norm(pts3d,dim=2)
    pts3d = (pts3d - pts3d.min()) / (pts3d.max() - pts3d.min())

    mask = torch.zeros_like(pts3d)
    mask[pts3d < cut_off] = 1

    return mask

def weighted_conf(conf,weight):
    return (conf - 1) * weight + 1
