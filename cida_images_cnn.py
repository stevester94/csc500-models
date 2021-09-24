#! /usr/bin/env python3

import torch.nn as nn
import torch
import torch.functional as F



def init_weight_STN(stn):
    """ Initialize the weights/bias with (nearly) identity transformation
    reference: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """
    stn[-5].weight[:, -28 * 28:].data.zero_()
    stn[-5].bias.data.zero_()
    stn[-1].weight.data.zero_()
    stn[-1].bias.data.copy_(torch.tensor([1 - 1e-2, 1e-2, 1 - 1e-2], dtype=torch.float))


def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)

    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 3:
        A_dim = 2
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim, A_dim)
    A = A_vec.new_zeros((A_vec.shape[0], A_dim, A_dim))
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()

class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class nnUnsqueeze(nn.Module):
    def __init__(self):
        super(nnUnsqueeze, self).__init__()

    def forward(self, x):
        return x[:, :, None, None]

class DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderSTN(nn.Module):
    def __init__(self):
        super(EncoderSTN, self).__init__()

        nh = 256
        nz = 100
        DROPOUT = 0.2
        DIM_DOMAIN = 1

        self.fc_stn = nn.Sequential(
            nn.Linear(DIM_DOMAIN + 28 * 28, nh), nn.LeakyReLU(0.2), nn.Dropout(DROPOUT),
            nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.LeakyReLU(0.2), nn.Dropout(DROPOUT),
            nn.Linear(nh, 3),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(DROPOUT),  # 14 x 14
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(DROPOUT),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(DROPOUT),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),
            nn.Linear(nh, 10),
        )

    def stn(self, x, u):
        # A_vec = self.fc_stn(u)
        A_vec = self.fc_stn(torch.cat([u, x.reshape(-1, 28 * 28)], 1))
        A = convert_Avec_to_A(A_vec)
        _, evs = torch.symeig(A, eigenvectors=True)
        tcos, tsin = evs[:, 0:1, 0:1], evs[:, 1:2, 0:1]

        self.theta_angle = torch.atan2(tsin[:, 0, 0], tcos[:, 0, 0])

        # clock-wise rotate theta
        theta_0 = torch.cat([tcos, tsin, tcos * 0], 2)
        theta_1 = torch.cat([-tsin, tcos, tcos * 0], 2)
        theta = torch.cat([theta_0, theta_1], 1)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x, u):
        """
        :param x: B x 1 x 28 x 28
        :param u: B x nu
        :return:
        """
        x = self.stn(x, u)
        z = self.conv(x)
        y = self.fc_pred(z)
        return F.log_softmax(y, dim=1), x, z


from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CIDA_Images_CNN_Model(nn.Module):
    def __init__(self,
        num_output_classes):
        super(CIDA_Images_CNN_Model, self).__init__()

        nz = 100
        dim_domain = 1

        self.netE = EncoderSTN()
        self.init_weight(self.netE)
        self.netD = DiscConv(nz, nout=dim_domain)

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x, t, alpha):
        y_hat, _, z = self.netE(x,t)

        reverse_z = ReverseLayerF.apply(z, alpha)

        t_hat = self.netD(reverse_z)

        return y_hat, t_hat