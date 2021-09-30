#! /usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Function



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
        # SM: Had to reshape u because the original code assumed it as in a -1,1 tensor
        reshaped_u = u.reshape(-1,1)
        reshaped_x = x.reshape(-1, 28 * 28)
        A_vec = self.fc_stn(torch.cat([reshaped_u, reshaped_x], 1))


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
    def __init__(
            self,
            num_output_classes,
            label_loss_object,
            domain_loss_object,
            learning_rate
        ):
        super(CIDA_Images_CNN_Model, self).__init__()

        self.label_loss_object = label_loss_object
        self.domain_loss_object = domain_loss_object

        nz = 100
        dim_domain = 1

        self.netE = EncoderSTN()
        self.init_weight(self.netE)
        self.netD = DiscConv(nz, nout=dim_domain)

        # self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=learning_rate)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=learning_rate)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))

        # Cargo cult
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


    # def forward(self, x, u):
    #     y_hat, _, z = self.netE(x,u)
    #     u_hat = self.netD(z)
    #     u_hat = u_hat.reshape(-1)

    #     return y_hat, u_hat

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.f, self.x_align, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.f.detach(), dim=1)

    def backward_G(self):
        self.d = self.netD(self.e).reshape(-1) # SM: Flatten it

        E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        E_gan_tgt = F.mse_loss(self.d[self.is_source == 0], self.u[self.is_source == 0])
        self.loss_E_gan = - (E_gan_src + E_gan_tgt) / 2

        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source)

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d = self.netD(self.e.detach()).reshape(-1) # SM: Flatten it

        D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        D_tgt = F.mse_loss(self.d[self.is_source == 0], self.u[self.is_source == 0])
        self.loss_D = (D_src + D_tgt) / 2
        self.loss_D.backward()

        self.D_src = D_src
        self.D_tgt = D_tgt

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def learn(self, x,y,u, is_source, alpha):
        self.x = x
        self.y = y
        self.u = u.float()
        self.is_source = is_source
        self.lambda_gan = alpha

        self.optimize_parameters()

        learn_results = {}
        learn_results["source_label_loss"] = self.loss_E_pred
        learn_results["source_domain_loss"] = self.D_src
        learn_results["target_domain_loss"] = self.D_tgt

        return learn_results
    
    def test(self, x,y,u):
        pass
    # def learn(self, x,y,u,s, alpha, domain_only:bool):
    #     """
    #     u is the domain vector
    #     s is the "is source?" vector
    #     """

    #     """
    #     returns a dict of
    #     {
    #         label_loss:float, # if domain_only==False
    #         domain_loss:float
    #     }
    #     """

    #     x_src = x[s]
    #     y_src = y[s]
    #     u_src = u[s]

    #     # Do domain's loss first, it is straight forward
    #     y_hat_src, u_hat_src = self.forward(x_src,u_src)
    #     descriminator_loss = self.domain_loss_object(u_hat, u_src)
    #     self.set_requires_grad(self.netD, True)
    #     self.optimizer_D.zero_grad()
    #     descriminator_loss.backward()
    #     self.optimizer_D.step()

    #     # Do encoder and potentially predictor. Note the negation of the descriminator loss
    #     y_hat, u_hat = self.forward(x,u) # Yeah it's dumb but I can't find an easy way to train the two nets separately without this
    #     encoder_loss = - alpha * self.domain_loss_object(u_hat, u)


    #     if not domain_only:
    #         label_loss = self.label_loss_object(y_hat, y)
    #         encoder_loss += label_loss

    #     # TODO: Disable Encoder Learning
    #     self.set_requires_grad(self.netD, False)
    #     self.optimizer_G.zero_grad()
    #     encoder_loss.backward()
    #     self.optimizer_G.step()


    #     if domain_only:
    #         return {
    #             "domain_loss": descriminator_loss
    #         }
    #     else:
    #         return {
    #             "domain_loss": descriminator_loss,
    #             "label_loss": label_loss
    #         }