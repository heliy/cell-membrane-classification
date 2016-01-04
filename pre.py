#coding: UTF-8

'''
preprocessing for image data
'''


import sys
import time
import random
from multiprocessing import Pool, Array

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata

# import theano
# import theano.tensor as T
# from theano import function as F
# from theano.ifelse import ifelse
# from theano.tensor.signal.conv import conv2d

# data
trLabels = np.load('data/train-labels.npy')
trVolume = np.load('data/train-volume.npy')
teVolume = np.load('data/test-volume.npy')

def expend(mats, window):
    '''
    Mirroring images with a size of window

    Parameters
    ----------
    mats : numpy array, shape (K, R, C)
         the mats of K images, with shape is R x C
    window : int
         the size of window to mirror

    Returns:
    ----------
    return_value : numpy array, shape (K, R+2*window, C+2*window)
    '''
    n, row, col = mats.shape
    reverse = range(window)[::-1]
    grounds = np.zeros((n, row+window*2, col+window*2)).astype('int')

    @np.vectorize
    def func(i):
        grounds[i, :window, :window] = np.rot90(mats[i, 1:window+1, 1:window+1], 2)
        grounds[i, :window, window:window+col] = mats[i, 1:window+1, :][reverse]
        grounds[i, :window, window+col:] = np.rot90(mats[i, 1:window+1, col-window-1:-1], 2)
    
        grounds[i, window:window+row, :window] = mats[i, :, 1:window+1][:, reverse]
        grounds[i, window:window+row, window:window+col] = mats[i, :, :]
        grounds[i, window:window+row, window+col:] = mats[i, :, col-window-1:-1][:, reverse]
    
        grounds[i, window+row:, :window] = np.rot90(mats[i, row-window-1:-1, 1:window+1], 2)
        grounds[i, window+row:, window:window+col] = mats[i, row-window-1:-1, :][reverse]
        grounds[i, window+row:, window+col:] = np.rot90(mats[i, row-window-1:-1, col-window-1:-1], 2)

    func(range(n))    
    return grounds

def random_rotate(mats):
    '''
    Rotate images randomly
    1/4 +90 ; 1/4 +180; 1/4 +270

    Parameters
    ---------
    mats : numpy array, shape(K, N, N)
         the mats of K images, with shape is N x N

    Returns:
    ---------
    return_value : numpy array, shape(K, N, N)
    '''
    assert len(mats.shape) == 3 and mats.shape[1] == mats.shape[2]
    ns = np.empty(mats.shape)
    @np.vectorize
    def func(i, r):
        ns[i] = np.rot90(mats[i], r)

    func(range(mats.shape[0]), np.random.randint(0, 3, mats.shape[0]))
    return ns

# center size for doing NO filtering
center = 4

def gauss2D(shape=(3,3),sigma=0.5):
    '''
    Get 2D Gaussian Distribution

    Parameters
    ---------
    shape : int tuple, default as (3, 3)
         the shape of Gaussian distribution to return
    sigma : float, default as 0.5
         the sigma parameter for Gaussian distribution

    Returns
    ---------
    gaussian_array : numpy array, shape as specified
    '''
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filters(window_size=95):
    '''
    Get batch filters with specified window size.
    Those filters considered the shape in different postion, and can be used in foveation directly.

    Parameters
    ---------
    window_size : int, default as 95
        the size of images used to be foveated

    Returns
    ---------
    filters : numpy array, shape is (window_size, window_size, max_size, max_size)
        every (i, j) in filters[i, j] is a filter to do Gaussian smooth in this location
        max_size is the max size of all filters, depended on window_size
    '''
    half = window_size//2
    max_size = ((window_size-half)//(2*center))*2+1
    filters = np.zeros((window_size, window_size, max_size, max_size))
    for i in range(window_size):
        for j in range(window_size):
            sigma = ((5*max(abs(i-half), abs(j-half)))/half)+0.001
            size = max(abs(i-half)//(2*center), abs(j-half)//(2*center))*2+1
            edge = (max_size - size)//2
            filters[i, j, edge:edge+size, edge:edge+size] = gauss2D((size, size), sigma)
    return filters

def foveate(mat, filters):
    '''
    A Orginal version for foveation
    vectorized and not run as matrix function
    We NOT use it in final version
    '''
    window_size = mat.shape[0]
    foveated = mat.copy()
    half = window_size//2
    fix_point = np.array(mat.shape)//2
    mirror = expend(mat, half)

    @np.vectorize
    def func(i, j):
        ds = np.abs(np.array([i, j]) - fix_point).astype('int')
        if ds[0] < center and ds[1] < center:
            return
        filter = filters[ds[0]][ds[1]]
        size = filter.shape[0]//2
        m = mirror[half+i-size:half+i+size+1, half+j-size:half+j+size+1]
        foveated[i, j] = int(round((m*filter).sum()))

        
    X, Y = np.meshgrid(range(window_size), range(window_size))
    func(X, Y)
    return foveated
    
def sampling_function(ground, window_size, ratio=1.5):
    '''
    Get a function for nonuniform sampling
    implemented with scipy.interpolate.griddata
    We NOT use it in final version    
    '''
    window_large = int(window_size*ratio//2)*2
    X, Y = np.meshgrid(np.linspace(-1, 1, window_large), np.linspace(-1, 1, window_large))
    X_, Y_ = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
    X_, Y_ = (3*X_**3+X_)/4, (3*Y_**3+Y_)/4
    x = np.dstack((X, Y)).reshape((window_large**2, 2))
    xi = np.dstack((X_, Y_)).reshape((window_size**2, 2))
    window = window_large//2

    def func(i, j):
        m = ground[i+window_size-window:i+window_size+window, j+window_size-window:j+window_size+window]
        return griddata(x, m.flat, xi, method='cubic').reshape((window_size, window_size)).astype('int')
    return func

def crop(size, points, grounds):
    '''
    Crop images in images

    Parameters
    --------
    size : int,
        the size of croped images
    points : (array of x, array of y)
        location of points which are specified to croping images
    grounds : numpy array, shape is (K, R, C)
        ground images to be cropped
        if specified location as (i, j) to be cropped, it will crop all K images

    Returns
    --------
    mats : numpy array, shape (N, size, size)
        N is K times the number of specified points
    '''
    l = grounds.shape[0]
    half_size = size//2
    xs, ys = points
    mats = np.zeros((xs.shape[0]*l, size, size))
        
    @np.vectorize
    def crop_points(n, i, j):
        mats[n*l:(n+1)*l, :, :] = grounds[:, i-half_size:i+half_size+1, j-half_size:j+half_size+1]
        
    crop_points(range(xs.shape[0]), xs, ys)
    return mats

def nonuni_sampling_template(from_size, to_size, subs_num=16, reserved=10):
    '''
    Get the sampling template of nonuniform sampling

    Parameters
    -------
    from_size : int
        the size of original image, must be a ODD integer
    to_size : int
        the size of samplied image, must be a ODD integer
    subs_num : int, defualt as 16
        the number of subs to split original image into subarranges, must be an EVEN integer
    reserved : int, defualt as 10
        the number of subs to reserved as unchanged in template, must be asb EVEN integer

    Returns
    -------
    template : numpy array, shape is (M,)
        M is the size of choosed points

    Notes
    -------
    Once you got the template
    ``` temp = nonuni_sampling_template(160, 120) ```
    You should use it like
    ``` samplied = img.flatten()[temp].reshape((120, 120)) ```
    '''
    assert from_size%2 == 1
    assert to_size%2 == 1
    assert subs_num%2 == 0
    assert reserved%2 == 0
    from_half, to_half = from_size//2, to_size//2
    subs_num = subs_num//2
    subs_size = from_half//subs_num+1
    reserved_range = np.arange(from_half-subs_size*reserved//2, from_half+subs_size*reserved//2+1)
    sampling_half_size = (to_size-reserved_range.shape[0])//2
    
    sampling_subs_num = (reserved_range[0])//subs_size+1
    p = np.array([[(sampling_subs_num+i)**2]*subs_size for i in range(sampling_subs_num)]).flatten()[:reserved_range[0]]
    p = p/p.sum()
    up = np.random.choice(np.arange(reserved_range[0]), size=sampling_half_size, replace=False, p=p)
    up.sort()
    p = np.array([[(3*sampling_subs_num-i)**2]*subs_size for i in range(sampling_subs_num)]).flatten()[:reserved_range[0]]
    p = p/p.sum()
    down = np.random.choice(np.arange(reserved_range[-1]+1, from_size), size=sampling_half_size, replace=False, p=p)
    down.sort()
    selected = np.concatenate((up, reserved_range, down))
    A = np.arange(from_size**2).reshape((from_size, from_size))
    a = A[selected, :]
    a = a[:, selected]
    return a

def template_sampling(grounds, to_size):
    '''
    Nonuniform sampling for images

    Parameters
    ------
    grounds : numpy array, shape (K, from_size, from_size)
        K images for sampling
    to_size : int
        specified size to sampling

    Returns
    ------
    mats : numpy array, shape (K, to_size, to_size)
    '''
    from_size = grounds.shape[1]
    template = nonuni_sampling_template(from_size, to_size, 20)
    mats = np.zeros((grounds.shape[0], to_size, to_size))

    @np.vectorize
    def func(i):
        mats[i] = grounds[i].flatten()[template].reshape((to_size, to_size))

    func(range(grounds.shape[0]))
    return mats    

def batch_foveate(mats, filts):
    '''
    vectorized matrix method for foveation

    Parameters
    -------
    mats : numpy array, shape (K, w, w)
        K images with shape is (w, w)
    filts : numpy array, shape (W, W, E, E)
        W x W filter with max_size is E
        assert w == (W+2*E)

    Returns
    -------
    foveated : numpy array, shape(K, W, W)
    '''
    edge = filts.shape[2]//2
    num = mats.shape[0]
    window = mats.shape[1]-2*edge

    filtereds = np.zeros((num, window, window))
    filter_pis = (edge*2+1)**2

    # tried to using Theano, but this method is slower than vectorized
    # if use_theano:
    #     for i in range(window):
    #         for j in range(window):
    #             m = theano.shared(mats[:, i:i+2*edge+1, j:j+2*edge+1].reshape((num, filter_pis)))
    #             filter = theano.shared(filts[i, j].reshape((filter_pis, 1)))
    #             filtereds[:, i, j] = T.dot(m, filter).eval().astype('int').flatten()
    # else:
    
    @np.vectorize
    def func(i, j):
        m = mats[:, i:i+2*edge+1, j:j+2*edge+1].reshape((num, filter_pis))
        f = filts[i, j].reshape((filter_pis, 1))
        filtereds[:, i, j] = np.dot(m, f).astype('int').flatten()
        
    X, Y = np.meshgrid(range(window), range(window))
    func(X, Y)
            
    return filtereds

def get_ys(labels, points):
    '''
    Get labels in points
    '''
    pages_num = labels.shape[0]
    Y = np.zeros((points[0].shape[0]*pages_num, 2))

    @np.vectorize
    def func(i, x, y):
        ys = [labels[l, x, y]== 0 and [1, 0] or [0, 1]for l in range(pages_num)]
        Y[i*pages_num:(i+1)*pages_num] = np.array(ys).reshape((pages_num, 2))

    func(range(points[0].shape[0]), points[0], points[1])
    return Y

def batch(prefix="train", volumes=trVolume, labels=trLabels, window_size=95, batch_size=30720, ratio=0.3, alreadyget=0, do_any=True):
    ''' current preprocessing method '''
    begin = time.time()
    print("begin:", begin)
    
    assert window_size%2 == 1
    page_num = volumes.shape[0]
    assert batch_size%page_num == 0
    fils = filters(window_size)
    filter_edge = fils.shape[3]
    grounds = expend(volumes, window_size)

    selected_x, selected_y = np.where(np.random.rand(volumes.shape[1], volumes.shape[2]) < ratio)
    batch_num = (selected_x.shape[0]*page_num)//batch_size
    pages_batch_size = batch_size//page_num
    hasLabel = labels is not None
    print("total", batch_num, "batches")
    
    # def func(batch_no):
    for batch_no in range(alreadyget, batch_num):
        print("batch", batch_no, ": ")
        points_x = selected_x[batch_no*pages_batch_size:(batch_no+1)*pages_batch_size]
        points_y = selected_y[batch_no*pages_batch_size:(batch_no+1)*pages_batch_size]
        if do_any:
            mats = crop(window_size*2+1, (points_x+window_size, points_y+window_size), grounds)
        else:
            mats = crop(window_size, (points_x+window_size, points_y+window_size), grounds)
        if do_any:
            print("nonuniform sampling ...")
            print(time.time())
            mats = template_sampling(mats, window_size+filter_edge-1)
            print("foveate ...")
            print(time.time())
            mats = batch_foveate(mats, fils)
        print("rotate ...")
        print(time.time())
        mats = random_rotate(mats)
        name = "%s%s_%d_%d_%d_" % (dir_prefix, prefix, window_size, batch_size, batch_no)
        print("save in", name)
        print(time.time())
        np.save(name+"x", mats)
        if hasLabel:
            ys = get_ys(labels, (points_x, points_y))
            np.save(name+"y", ys)
        print(time.time())

    # with Pool(processes=processes) as pool:
        # pool.starmap(func, range(batch_num))
        
    print("begin:", begin)
    print("end:", time.time())
        

def append_sample(prefix="test255-2", LABEL=255, window_size=95, batch_size=3000, ratio=0.3, begin=500, hajimari=0, do_any=True):
    ''' extract more membrane samples'''
    assert window_size%2 == 1
    page_num = 1
    assert batch_size%page_num == 0
    fils = filters(window_size)
    filter_edge = fils.shape[3]
    grounds = expend(trVolume, window_size)

    batch = -1
    for (volume, label, ground) in zip(trVolume, trLabels, grounds):
        mems = np.arange(volume.shape[0]*volume.shape[1]).flatten()[(label == LABEL).flatten()]
        selected_points = mems[np.random.rand(mems.shape[0]) < ratio]
        print(mems.shape, selected_points.shape)
        selected_x = (selected_points/volume.shape[0]).astype("int")
        selected_y = (selected_points%volume.shape[0]).astype('int')
        batch_num = (selected_x.shape[0])//batch_size+1
        print("total", batch_num, "batches")
        for batch_no in range(batch_num):
            batch += 1
            if batch < hajimari:
                continue
            print("batch", batch, ": ")
            points_x = selected_x[batch_no*batch_size:(batch_no+1)*batch_size]
            points_y = selected_y[batch_no*batch_size:(batch_no+1)*batch_size]
            print("cropping ...")
            print(time.time())
            if do_any:
                mats = crop(window_size*2+1, (points_x+window_size, points_y+window_size), np.array([ground]))
            else:
                mats = crop(window_size, (points_x+window_size, points_y+window_size), np.array([ground]))
            if do_any:
                print("nonuniform sampling ...")
                print(time.time())
                mats = template_sampling(mats, window_size+filter_edge-1)
                print("foveate ...")
                print(time.time())
                mats = batch_foveate(mats, fils)
            print("rotate ...")
            print(time.time())
            mats = random_rotate(mats)
            name = "%s%s_%d_%d_%d_" % (dir_prefix, prefix, window_size, batch_size, batch+begin)
            print("save in", name)
            print(time.time())
            np.save(name+"x", mats)
        print(time.time())
        
    
# test: batch_size=7680
# train 95: already get 9
    
def batch_func(prefix, no, volumes, labels, window_size, batch_size, ratio, sampling_ratio):
    fils = filters(window_size)
    store_x = np.zeros((batch_size, window_size, window_size))
    store_y = np.zeros((batch_size, 2))
    ground = expend(volumes, window_size)
    sampling = sampling_function(ground, window_size, sampling_ratio)
    hasLabel = labels is not None

    @np.vectorize
    def func(i, j, cur):
        mat = sampling(i, j).astype('int')
        store_x[cur] = random_rotate(foveate(mat, fils))
        if cur%50 == 0:
            print(cur)
        if hasLabel:
            store_y[cur] = labels[i, j] == 0 and [1, 0] or [0, 1]

    print(time.time())
    X, Y = np.where(np.random.rand(labels.shape[0], labels.shape[1]) < ratio)
    for batch_num in range(X.shape[0]//batch_size):
        func(X[batch_num*batch_size: (batch_num+1)*batch_size],
             Y[batch_num*batch_size: (batch_num+1)*batch_size],
             range(batch_size))
        name = 'data/tmp/%s_%d_%d_%d_' % (prefix, window_size, no, batch_num)
        np.save(name+"x", store_x)
        store_x.fill(0)
        if hasLabel:
            np.save(name+"y", store_y)
            store_y.fill(0)
        print(time.time())
        print("write file: ", name)
        print(batch_num, " batch is OVER ... ")
        
    return True

def map_batch(processes=4, window_size=95, batch_size=20000, ratio=0.3, sampling_ratio=2):
    l = trLabels.shape[0]
    with Pool(processes=processes) as pool:
        z = zip(np.repeat('train', l), range(l), trVolume, trLabels,
                                     np.repeat(window_size, l), np.repeat(batch_size, l),
                                     np.repeat(ratio, l), np.repeat(sampling_ratio, l))
        pool.starmap(batch_func, z)

# if __name__ == '__main__':
#     p, w = sys.argv[1:]
#     map_batch(int(p), int(w))
