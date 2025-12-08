import torch
import torch.nn as nn
from det_layer import DetLayer

class DetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = DetLayer(784, 256)
        self.l2 = DetLayer(256, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x, F_res=1.0):
        x, d1_loss, d1_debug = self.l1(x, F_res=F_res)
        x, d2_loss, d2_debug = self.l2(x, F_res=F_res)
        logits = self.out(x)

        det_loss = d1_loss + d2_loss
        debug = {"l1": d1_debug, "l2": d2_debug}
        return logits, det_loss, debug