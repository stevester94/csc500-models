#! /usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Function



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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        nh = 256
        nz = 100
        DROPOUT = 0.2
        DIM_DOMAIN = 1

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

    def forward(self, x, u):
        """
        :param x: B x 1 x 28 x 28
        :param u: B x nu
        :return:
        """
        z = self.conv(x)
        y = self.fc_pred(z)
        return F.log_softmax(y, dim=1), x, z





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

        self.netE = Encoder()
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


    def forward(self, x, u):
        y_hat, _, z = self.netE(x,u)
        u_hat = self.netD(z)
        u_hat = u_hat.reshape(-1)

        return y_hat, u_hat

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def learn(self, x,y,u, alpha, domain_only:bool):
        """
        returns a dict of
        {
            label_loss:float, # if domain_only==False
            domain_loss:float
        }
        """

        # Do domain's loss first, it is straight forward
        y_hat, u_hat = self.forward(x,u)
        descriminator_loss = self.domain_loss_object(u_hat, u)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        descriminator_loss.backward()
        self.optimizer_D.step()

        # Do encoder and potentially predictor. Note the negation of the descriminator loss
        y_hat, u_hat = self.forward(x,u) # Yeah it's dumb but I can't find an easy way to train the two nets separately without this
        encoder_loss = - alpha * self.domain_loss_object(u_hat, u)


        if not domain_only:
            label_loss = self.label_loss_object(y_hat, y)
            encoder_loss += label_loss

        # TODO: Disable Encoder Learning
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        encoder_loss.backward()
        self.optimizer_G.step()


        if domain_only:
            return {
                "domain_loss": descriminator_loss
            }
        else:
            return {
                "domain_loss": descriminator_loss,
                "label_loss": label_loss
            }
