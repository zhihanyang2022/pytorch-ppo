import os
import torch
import torch.nn as nn


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    torch.save(net.state_dict(), os.path.join(save_dir, save_name))


def load_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    net.load_state_dict(
        torch.load(os.path.join(save_dir, save_name), map_location=torch.device(get_device()))
    )
