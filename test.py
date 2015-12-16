#coding:UTF-8

import cv2

from data import trVolume, filters, foveate, sampling_function, expend

def test_fovea():
    img = trVolume[0, :511, :511]
    fis, sigs = filters(img.shape[0])
    fov = foveate(img, fis)
    # return fov
    cv2.imshow('image', fov)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sigs, fov

def test_sampling():
    window_size = 512
    g = expend(trVolume[0, :, :], window_size).astype('int')
    sampling = sampling_function(g, window_size, 2)
    img = sampling(0, 0)
    cv2.imwrite('data/sa.png', img)
    return g
    
