#coding:UTF-8

import cv2

from pre import trVolume, filters, foveate, template_sampling, expend

import numpy as np

def test_fovea():
    img = trVolume[0, :511, :511]
    fis = filters(img.shape[0])
    fov = foveate(img, fis)
    # return fov
    cv2.imshow('image1', fov)
    cv2.imshow('image2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return fov, img

def test_sampling():
    img = template_sampling(np.array([trVolume[0, :511, :511]]), 301)
    cv2.imwrite('data/sa.png', img[0])
    return img[0]
    
