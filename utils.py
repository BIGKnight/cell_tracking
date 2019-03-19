from PIL import Image
import matplotlib.pyplot as plt
from math import *


def img_crop(mode):
    for i in range(39):
        img = Image.open("/home/zzn/PycharmProjects/cell_tracking/images/" + str(i) + ".png")
        if mode == 'origin':
            box = (0, 0, img.size[0] // 2, img.size[1])
            roi = img.crop(box)
            roi.save('/home/zzn/PycharmProjects/cell_tracking/images/origin/%d.png' % i, 'PNG')

        elif mode == 'segmentation':
            box = (img.size[0] // 2, 0, img.size[0], img.size[1])
            roi = img.crop(box)
            roi.save('/home/zzn/PycharmProjects/cell_tracking/images/segmentation/%d.png' % i, 'PNG')


def bilinear_interpolation(img, height, width, h, w):
    h_low = floor(h)
    w_low = floor(w)
    h_high = h_low + 1
    w_high = w_low + 1
    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = v2 = v3 = v4 = 0.
    if h_low >= 0 and w_low >=0:
        v1 = img[h_low][w_low]
    if h_low >= 0 and w_high <= width - 1:
        v2 = img[h_low][w_high]
    if h_high <= height -1 and w_low >= 0:
        v3 = img[h_high][w_low]
    if h_high <= height -1 and w_high <= width - 1:
        v4 = img[h_high][w_high]

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)
    return val
