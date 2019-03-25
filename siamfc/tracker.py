import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import warnings
import torchvision.transforms as transforms

from torch.autograd import Variable
from siamesenet import SiameseNet
from config import config
from custom_transforms import ToTensor
from utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image


class SiamFCTracker:
    def __init__(self, model_path, gpu_id):
        print(model_path)
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id):
            self.model = SiameseNet()
            # Since we created the model using nn.DataParallel, we use it
            # again to load the state_dict
            self.model.features = torch.nn.DataParallel(self.model.features)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
            self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(
            np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        # Get target initial bbox (zero based)
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0]-1 +
                     bbox[2], bbox[1]-1+bbox[3])  
        # Get target center x, center y, zero based
        self.pos = np.array([bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])
        print('Initial position ==>', self.pos)
        # Get target width, height
        self.target_sz = np.array([bbox[2], bbox[3]])
        print('Initial size ==>', self.target_sz)

        # Get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        print('\nCONTEXT AMOUNT ===========>', config.context_amount)

        # scale_z is the number by which we multiply the original image to fit it in a 127x127
        # new image with the target centered. It depends on the context amount.
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                                                        config.exemplar_size, config.context_amount, self.img_mean)
        print('scale_z ==>', scale_z)
        print('s_z ==>', s_z)
        # cv2.imwrite('/home/lv71186/deutsch/pysiamfc/dev/tracker_test/exemplar'+str(config.context_amount).replace('.','_')+'.jpg', exemplar_img)

        # Get exemplar feature map
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        with torch.cuda.device(self.gpu_id):
            exemplar_img_var = Variable(exemplar_img.cuda())
            self.model(exemplar_img_var, None)

        # Define penalties
        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1
        print('Penalty ==>', self.penalty)

        # Define Interpolation response size (by default 272x272). We upsample the original score map
        # (which is 17x17) so we can get more accurate localization.
        self.interp_response_sz = config.response_up_stride * config.response_sz

        # Define cosine window
        self.cosine_window = self._cosine_window(
            (self.interp_response_sz, self.interp_response_sz))
        print('Cosine window ==>', self.cosine_window.shape)

        # Create scales
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                                                     np.floor(config.num_scale/2)+1)
        print('Scales ==>', self.scales)
        # create s_x
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z
        print('s_x ==>', self.s_x)

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(
            frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        # for i, image in enumerate(pyramid):
            # cv2.imwrite('/home/lv71186/deutsch/pysiamfc/dev/tracker_test/pyramid-'+str(i)+'.jpg', image)
        instance_imgs = torch.cat(
            [self.transforms(x)[None, :, :, :] for x in pyramid], dim=0)
        
        # Get Score map for a batch of search images with different scales
        with torch.cuda.device(self.gpu_id):
            instance_imgs_var = Variable(instance_imgs.cuda())
            response_maps = self.model(None, instance_imgs_var)
            response_maps = response_maps.data.cpu().numpy().squeeze()

            # Upsamble the Score Map using Bicubic Interpolation theoretically results
            # in more accurate localization
            response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
                                for x in response_maps]

        # Get max score of the score map for every scale
        max_score = np.array([x.max()
                              for x in response_maps_up]) * self.penalty
        
        print('Max Score ==>', max_score)

        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        print('response map ==>', response_map.shape)
        response_map -= response_map.min()
        response_map /= response_map.sum()
        print('response map SUM ==>', response_map.sum())
        
        # Apply Cosine Window to penalize large displacements
        response_map = (1 - config.window_influence) * response_map + \
            config.window_influence * self.cosine_window
        
        # Get position of the highest value of the score map. argmax() give the index in the
        # flattened array (1D). unravel_index give the indexes of the unflattened version
        # according to the dimensions we specify with response_map.shape()
        max_r, max_c = np.unravel_index(
            response_map.argmax(), response_map.shape)
        print('response argmax ==>', response_map.argmax())
        print('max_r, max_C ==>', max_r, max_c)

        # displacement in interpolation response
        disp_response_interp = np.array(
            [max_c, max_r]) - (self.interp_response_sz-1) / 2.
        print('Interpolation displacement ==>', disp_response_interp)

        # displacement in input
        disp_response_input = disp_response_interp * \
            config.total_stride / config.response_up_stride
        print('Input displacement ==>', disp_response_input)

        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input * \
            (self.s_x * scale) / config.instance_size

        print('Frame displacement ==>', disp_response_frame)
        # position in frame coordinates
        self.pos += disp_response_frame
        print('Position ==>', self.pos)

        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) +
                          config.scale_lr * scale) * self.target_sz
        bbox = (self.pos[0] - self.target_sz[0]/2 + 1,  # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1]/2 + 1,  # ymin
                self.pos[0] + self.target_sz[0]/2 + 1,  # xmax
                self.pos[1] + self.target_sz[1]/2 + 1)  # ymax
        print('bbox ==>', self.bbox)
        return bbox
