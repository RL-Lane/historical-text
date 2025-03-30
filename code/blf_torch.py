# https://gist.github.com/etienne87/9f903b2b16389f9fe98a18fade6df74b

import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gkern2d(l=21, sig=3, device=DEVICE):
    """Returns a 2D Gaussian kernel array."""
    ax = torch.arange(-l // 2 + 1., l // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel


class Shift(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(Shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.channels_per_group = self.in_planes // (self.kernel_size ** 2)
        if self.kernel_size == 3:
            self.pad = 1
        elif self.kernel_size == 5:
            self.pad = 2
        elif self.kernel_size == 7:
            self.pad = 3

    def forward(self, x):
        n, c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        # Alias for convenience
        cpg = self.channels_per_group
        cat_layers = []
        for i in range(self.in_planes):
            # Parse in row-major
            for y in range(0,self.kernel_size):
                y2 = y+h
                for x in range(0, self.kernel_size):
                    x2 = x+w
                    xx = x_pad[:,i:i+1,y:y2,x:x2]
                    cat_layers += [xx]
        return torch.cat(cat_layers, 1)


class BilateralFilter(nn.Module):
    """BilateralFilter computes:
        If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(self, channels=3, k=7, height=480, width=640, sigma_space=5, sigma_color=0.1, device=DEVICE):
        super(BilateralFilter, self).__init__()

        # space gaussian kernel
        self.gw = gkern2d(k, sigma_space, device=device)

        self.g = torch.tile(self.gw.reshape(channels, k*k, 1, 1), (1, 1, height, width))
        # shift
        self.shift = Shift(channels, k)
        self.sigma_color = 2*sigma_color**2

        self.to(device=device)

    def forward(self, I):
        Is = self.shift(I).data
        Iex = I.expand(*Is.size())
        D = (Is-Iex)**2 # here we are actually missing some sum over groups of channels
        De = torch.exp(-D / self.sigma_color)
        Dd = De * self.g
        W_denom = torch.sum(Dd, dim=1)
        If = torch.sum(Dd*Is, dim=1) / W_denom
        return If





# if __name__ == '__main__':
#     from skimage import data as ski_data
#     from skimage.color import rgb2gray
#     import time

#     k = 5
#     device = DEVICE

#     img = ski_data.astronaut()
#     img = rgb2gray(img)
#     img = img[None, None, ...]
#     img = torch.tensor(img)
#     if device:
#         img = img.cuda()

#     bilat = BilateralFilter(img.shape[1], k, img.shape[2], img.shape[3], device=device)

#     start_time = time.time()
#     img_filtered = bilat(img)
#     print("Duration: ", time.time() - start_time)

#     img_out = img_filtered.cpu().numpy().squeeze()