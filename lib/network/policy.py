import torch.nn as nn
import torch.nn.functional as F

from lib import common

in_ch = 104
middle_ch = 192
out_ch = common.MOVE_DIRECTION_LABEL_NUM
ksize = 3


class PolicyNet(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch, ksize):
        super().__init__()
        self.in_ch = in_ch
        self.middle_ch = middle_ch
        self.out_ch = out_ch
        self.ksize = ksize
        self.conv01 = nn.Conv2d(in_ch, middle_ch, ksize, padding='same')
        self.conv02 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv03 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv04 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv05 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv06 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv07 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv08 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv09 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv10 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv11 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv12 = nn.Conv2d(middle_ch, middle_ch, ksize, padding='same')
        self.conv13 = nn.Conv2d(middle_ch, out_ch, 1)

    def forward(self, x):
        h01 = F.relu(self.conv01(x))
        h02 = F.relu(self.conv02(h01))
        h03 = F.relu(self.conv03(h02))
        h04 = F.relu(self.conv04(h03))
        h05 = F.relu(self.conv05(h04))
        h06 = F.relu(self.conv06(h05))
        h07 = F.relu(self.conv07(h06))
        h08 = F.relu(self.conv08(h07))
        h09 = F.relu(self.conv09(h08))
        h10 = F.relu(self.conv10(h09))
        h11 = F.relu(self.conv11(h10))
        h12 = F.relu(self.conv12(h11))
        h13 = self.conv13(h12)
        return h13.view(x.shape[0], -1)
