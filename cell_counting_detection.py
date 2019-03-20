import numpy as np
import torch
import torch.nn.functional as functional
import torch.nn as nn
from utils import bilinear_interpolation
import cv2
import math
import os


# third
class GVF_generator(nn.Module):
    def __init__(self):
        super(GVF_generator, self).__init__()
        self.sobel_x_filter = torch.FloatTensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3)
        self.sobel_y_filter = torch.FloatTensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)

    def forward(self, x):
        # x is a tensor, and it can only have oen channel
        x_gradient = functional.conv2d(x, self.sobel_x_filter, bias=None, stride=1, padding=1)
        y_gradient = functional.conv2d(x, self.sobel_y_filter, bias=None, stride=1, padding=1)
        gvf = torch.cat((x_gradient, y_gradient), dim=1)
        return gvf


# first
class Binarization(nn.Module):
    def __init__(self, dir):
        super(Binarization, self).__init__()
        maps = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for filepath in filenames:
                frame_path = os.path.join(dirpath, filepath)
                img = cv2.imread(frame_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = torch.FloatTensor(img_gray)
                maps.append(img_gray)
        self.binarization_maps = torch.stack(maps)

    def forward(self, alpha):
        threshold = 255 * alpha
        indicator_maps = 1 - self.binarization_maps.div(threshold).floor().clamp(0, 1)
        output = self.binarization_maps.mul(indicator_maps)
        image_shape = output.shape
        return output.view(image_shape[0], 1, image_shape[1], image_shape[2])


# second
class Gaussian_Filter(nn.Module):
    def __init__(self, sigma):
        super(Gaussian_Filter, self).__init__()
        self.sigma = sigma
        self.gaussian_radius = self.sigma
        self.gaussian_map = np.multiply(
            cv2.getGaussianKernel(self.gaussian_radius * 2 + 1, sigma),
            cv2.getGaussianKernel(self.gaussian_radius * 2 + 1, sigma).T
        )
        self.gaussian_map = torch.FloatTensor(self.gaussian_map).view(1, 1, self.gaussian_radius * 2 + 1, self.gaussian_radius * 2 + 1)

    def forward(self, x):
        output = functional.conv2d(x, self.gaussian_map, bias=None, stride=1, padding=self.gaussian_radius)
        return output


class Seeds_Searcher(nn.Module):
    def __init__(self, batch, height, width):
        super(Seeds_Searcher, self).__init__()
        self.batch = batch
        self.height = height
        self.width = width

    def forward(self, Mnuc, Mgvf):
        Mpar = torch.zeros(self.batch, self.height, self.width)
        nuc= [[] for i in range(self.batch)]
        for n in range(self.batch):
            for i in range(self.height):
                for j in range(self.width):
                    if Mnuc[n][0][i][j] > 0:
                        nuc[n].append((i, j))
        for index in range(len(nuc)):
            for i, j in nuc[index]:
                h0, w0 = (-1, -1)
                h1, w1 = (i, j)
                flag = 0
                count = 0
                while flag == 0 and count < 100 and (abs(h1 - h0) > 0.5 or abs(w1 - w0) > 0.5):
                    if h1 > 239 or h1 < 0 or w1 > 319 or w1 < 0:
                        flag = 1
                    else:
                        h_gradient = bilinear_interpolation(Mgvf[index][0], self.height, self.width, h1, w1)
                        w_gradient = bilinear_interpolation(Mgvf[index][1], self.height, self.width, h1, w1)
                        length = math.sqrt(h_gradient ** 2 + w_gradient ** 2)
                        if length > 1:
                            h_gradient = h_gradient / length
                            w_gradient = w_gradient / length
                        h0 = h1
                        w0 = w1
                        h1 = h1 + h_gradient
                        w1 = w1 + w_gradient
                        count += 1
                h1 = math.floor(h1)
                w1 = math.floor(w1)
                if flag == 0:
                    Mpar[index][h1][w1] += 1
        return Mpar


class Final_Filter_Module(nn.Module):
    def __init__(self, threshold):
        super(Final_Filter_Module, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        # indicator = x.div(self.threshold).floor().clamp(0, 1)
        # output = x * indicator
        coordinates_list = []
        for i in range(len(x)):
            counts = np.where(x[i] > self.threshold)
            coordinates_list.append(list(zip(counts[0], counts[1])))
        return coordinates_list


