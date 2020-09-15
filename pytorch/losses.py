import torch.nn.functional as F
import torch

def binary_cross_entropy(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])


def cross_entropy(output_dict, target_dict):
    targets = torch.argmax(target_dict['target'], dim=-1)
    return F.cross_entropy(output_dict['clipwise_output'], targets)


def get_loss_func(loss_type):
    if loss_type == 'binary_cross_entropy':
        return binary_cross_entropy
    if loss_type == 'cross_entropy':
        return cross_entropy