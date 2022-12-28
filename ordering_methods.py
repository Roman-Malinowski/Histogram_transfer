import numpy as np
import pandas as pd


def create_order_3_channels(img, img_smooth=None):
    """
    Creates an order on the pixels of img based on their 3 channels.
    The order is a tuple of two arrays of length img.shape[:2].product(),
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    The pixels are ordred by increasing 1st channel, then 2nd, then 3rd,
    then by increasing 1st, 2nd and 3rd channel of their neighbours.
    
    inputs:
    img: np.array. A 3-channel image, typically a HSV or BGR image obtained with cv.cvtColor(X, cv.COLOR_BGR2HSV)
    
    img_smooth: Optionnal. np.array. A smoothed version of img used for ranking pixels based on their neighbourhood
    Must have the same dimension as img (typically using cv.GaussianBlur(img, (5, 5), 0)).
    
    ouput:
    order: (np.array, np.array). An array img.shape[:2].product().
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    """
    
    indices = np.indices(img.shape[:2])
    
    if img_smooth is None:
        data=np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                              np.ravel(img[:,:,0]), np.ravel(img[:,:,1]), np.ravel(img[:,:,2])))

        df = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3"], data=data)
        df = df.sort_values(by=["ch1", "ch2", "ch3"])
        
    else:
        data=np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                              np.ravel(img[:,:,0]), np.ravel(img[:,:,1]), np.ravel(img[:,:,2]),
                              np.ravel(img_smooth[:,:,0]), np.ravel(img_smooth[:,:,1]), np.ravel(img_smooth[:,:,2])))
        
        df = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3", "smooth1", "smooth2", "smooth3"], data=data)
        df = df.sort_values(by=["ch1", "ch2", "ch3", "smooth1", "smooth2", "smooth3"])
    
    return (df["row"].to_numpy(dtype=int), df["col"].to_numpy(dtype=int))


def create_order_single_channel(img, img_smooth=None):
    """
    Creates an order on the pixels of img.
    The order is a tuple of two arrays of length img.shape[:2].product(),
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    The ordering is done by increasing value of the channel, and for
    same valued pixels, by the mean value of their neighbours.
    
    inputs:
    img: np.array. A 1-channel image, typically a greyscale image obtained with cv.cvtColor(X, cv.COLOR_BGR2GRAY)
    
    img_smoothed: np.array. Optionnal. A smoothed version of img used for ranking pixels based on their neighbourhood
    Must have the same dimension as img (typically using cv.GaussianBlur(img, (5, 5), 0)).
       
    ouput:
    order: (np.array, np.array). An array img.shape[:2].product().
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    """
    
    indices = np.indices(img.shape)
    
    if img_smooth is None:
        data=np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                              np.ravel(img)))

        df = pd.DataFrame(columns=["row", "col", "ch1"], data=data)
        df = df.sort_values(by=["ch1"])
        
    else:
        data=np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                              np.ravel(img), np.ravel(img_smooth)))
        
        df = pd.DataFrame(columns=["row", "col", "ch1", "smooth1"], data=data)
        df = df.sort_values(by=["ch1", "smooth1"])
    
    return (df["row"].to_numpy(dtype=int), df["col"].to_numpy(dtype=int))


def dist_RGB(r, g, b):
    return np.sqrt(np.square(r) + np.square(g) + np.square(b))


def transfer_with_dependence(img, img_smooth, img_ref, dist=dist_RGB):
    