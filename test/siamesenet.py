import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from config import config


class SiameseNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(SiameseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, groups=2)
        )
        np.set_printoptions(threshold=np.inf)
        torch.set_printoptions(profile="full")
        self.corr_bias = nn.Parameter(torch.zeros(1))
        gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))
        self.gt = torch.from_numpy(gt).cuda()
        self.weight = torch.from_numpy(weight).cuda()
        self.batch_normalization = nn.BatchNorm2d(1)

    def init_weights(self):
        """ Initializes the weights of the CNN model
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)
                nn.init.constant_(module.bias.data, 0.1)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, mean=1)
                module.bias.data.zero_()

    def forward(self, z, x):
        """Cross-correlation between the two feature maps extracted
        from search and reference images.
        
        Arguments:
            z {torch.Tensor} -- Reference Image samples
            x {torch.Tensor} -- Search Image samples
        
        Returns:
            [torch.Tensor] -- Score Map of the correlation
        """

        # print('SEARCH IMAGE ==>', x.shape)
        # print('REFERENCE IMAGE ==>', z.shape)
        search = self.features(x)
        # print('SEARCH MAP ==>', search.shape)
        reference = self.features(z)
        # print('MAX, MIN ==>', reference.max(), reference.min())
        # print('REFRERENCE MAP ==>', reference)
        N, C, H, W = search.shape
        search = search.view(1, -1, H, W)
        score_map = F.conv2d(search, reference, groups=N) * config.response_scale + self.corr_bias
        # score_map = self.batch_normalization(score_map)

        # print('score_map.transpose ==>', score_map.transpose(0,1))
        # print('score_map MAX, MIN ==>', score_map.max().item(), score_map.min().item())
        return score_map.transpose(0, 1)
    
    def loss(self, output):
        """Since we compute the loss for every mini-batch, we first calculate
        the loss for each sample, sum it and divide it by the batch size.
        
        Arguments:
            output {torch.Tensor} -- output of the network
        
        Returns:
            [type] -- [description]
        """

        # print('GT ==>', self.gt.shape)
        # print('SCORE MAP ==>', output.shape)
        # return F.binary_cross_entropy_with_logits(output, self.gt, self.weight, reduction='sum') / config.train_batch_size
        # print("BINARY LOSS ==>", binary.item())
        return F.soft_margin_loss(output, self.gt, reduction='sum') / config.train_batch_size

    def _create_gt_mask(self, shape):
        """Creates the Ground Truth Score Map to compute the loss
        
        Arguments:
            shape {tuple} -- Dimensions of Ground Truth mask
        
        Returns:
            [mask, weight] -- 
        """

        # same for all pairs
        h, w = shape

        # Creates a circle depending on network stride
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)

        # mask = np.zeros((h, w))
        mask = np.full((h, w), -1)
        mask[dist <= config.radius / config.total_stride] = 1

        # np.newaxis is used to increase the dimension 
        # of the existing array by one more dimension, when used once
        mask = mask[np.newaxis, :, :]
        
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == -1] = 0.5 / np.sum(mask == -1)

        # mask output size:
        # [batch_size, 1, config.response_size, config.response_size]
        # i.e., by default, [8,1,17,17]
        mask = np.repeat(mask, config.train_batch_size, axis=0)[
            :, np.newaxis, :, :]
        # print('Mask ==>', mask.shape)
        # print('weights ==>', weights)        
        return mask.astype(np.float32), weights.astype(np.float32)
