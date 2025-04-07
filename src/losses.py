import torch # type: ignore
import torch.nn # type: ignore
import numpy as np # type: ignore
import utils

METHOD_TYPE = 'supervised'

def calc_xent_loss_base2(logits, y, reduction='sum'):
    xent_base_e = torch.nn.functional.cross_entropy(logits, y, reduction=reduction)
    xent_base_2 = xent_base_e / np.log(2.0)
    return xent_base_2

def calc_labeled_loss_for_batch(logits_BC, y_B):
    xent_loss_per_example = calc_xent_loss_base2(logits_BC, y_B, reduction='mean')
    return xent_loss_per_example

def calc_xent_loss_with_l2(logits_BC, y_B, model, l2pen_mag, batch_size, reduction='mean'):
    ce_loss = calc_xent_loss_base2(logits_BC, y_B, reduction=reduction)
    l2_loss = 0.0
    if l2pen_mag > 0:
        params = utils.flatten_params(model)
        l2_loss = l2pen_mag * torch.sum(params ** 2) / batch_size
    return ce_loss + l2_loss
