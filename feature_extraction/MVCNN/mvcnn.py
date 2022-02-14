import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, model, pretrained, emb_dim, n_class):
        super(Model, self).__init__()
        _feat_shape = {'alexnet': 256 * 6 * 6, 'vgg': 512 * 7 * 7, 'resnet': 2048}
        self.encoder = None
        if model == 'alexnet':
            self.encoder = models.alexnet(pretrained=pretrained).features
        elif model == 'resnet':
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder.fc = nn.Linear(512, 2048)
        elif model == 'vgg':
            self.encoder = models.vgg19_bn(pretrained=pretrained).features
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(_feat_shape[model], 2048),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(),
            nn.Linear(2048, emb_dim),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(emb_dim, 2048),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(2048, n_class),
        )

    def forward(self, x):
        batch_size, n_view, __, __, __ = x.size()
        x = x.flatten(0, 1)
        x = self.encoder(x)
        x = x.flatten(1, -1)
        xi = self.head(x)
        xm = xi.view(batch_size, n_view, -1)
        xm, __ = torch.max(xm, 1)
        y = self.classifier(xm)
        return xi, xm, y

    @staticmethod
    def jsd_loss(encodingA, encodingB, indices):
        encodingA = F.normalize(encodingA, 2, 1)
        encodingB = F.normalize(encodingB, 2, 1)
        pos_mask = torch.eye(encodingA.shape[0], encodingB.shape[0], device=encodingA.device)
        pos_mask = pos_mask[indices]
        neg_mask = 1. - pos_mask
        logits = torch.mm(encodingA, encodingB.t())
        Epos = np.log(2.) - F.softplus(- logits)
        Eneg = F.softplus(-logits) + logits - np.log(2.)
        Epos = (Epos * pos_mask).sum() / pos_mask.sum()
        Eneg = (Eneg * neg_mask).sum() / neg_mask.sum()
        return Eneg - Epos

    @staticmethod
    def ce_loss(yp, yt, weights):
        return nn.CrossEntropyLoss(weight=weights)(yp, yt)

