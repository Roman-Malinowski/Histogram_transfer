import numpy as np


def create_order_3_channels(img, img_smooth=None, complete=False):
    """
    Creates an order on the pixels of img based on their 3 channels.
    The order is a tuple of two arrays of length img.shape[:2].product(),
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    The pixels are ordred by increasing 1st channel, then 2nd, then 3rd,
    then by increasing 1st, 2nd and 3rd channel of their neighbours.
    
    inputs:
    img: np.array. A 3-channel image, typically a HSV or BGR image obtained with cv.cvtColor(X, cv.COLOR_BGR2HSV)
    
    img_smooth: np.array. Optionnal. A smoothed version of img used for ranking pixels based on their neighbourhood
    Must have the same dimension as img (typically using cv.GaussianBlur(img, (5, 5), 0)). Required if complete=True
    
    complete: bool. Optionnal. If complete=False, then the order will be arbitrary for pixels with same values of (HSV)
    If complete=True, the order will rank pixels with same values based on the mean value of their neighbours.    
    
    ouput:
    order: (np.array, np.array). An array img.shape[:2].product().
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    """
    
    img_structured = img.copy()
    img_structured.dtype = np.dtype([('H', np.uint8), ('S', np.uint8), ('V', np.uint8)]) 
    """
    [[[[(3, 5, 5)]], [[(0, 1, 2)]]], [[[(3, 4, 5)]], [[(3, 6, 6)]]], [[[(6, 6, 8)]], [[(6, 6, 8)]]], [[[(4, 6, 1)]], [[(1, 0, 0)]]]]
    """
    n_cols = img_structured.shape[1]
    order = np.argsort(img_structured, axis=None, order=("H", "S", "V"))
    """
    [1 7 2 0 3 6 4 5]
    """
    if complete:
        
        assert img_smooth is not None, "For a complete ordering, there must be an img_smoothed given"
        
        img_smooth_structured = img_smooth.copy()
        img_smooth_structured.dtype = np.dtype([('H', np.uint8), ('S', np.uint8), ('V', np.uint8)])

        # Finding values that cannot be completely ordered with HSV
        values, counts = np.unique(img_structured, return_counts=True)
        """
        values = [(0, 1, 2), (1, 0, 0), (3, 4, 5), (3, 5, 5), (3, 6, 6), (4, 6, 1), (6, 6, 8)]
        count = [1, 1, 1, 1, 1, 1, 2]

        """
        tot = np.sum(counts>1)
        for i, val in enumerate(values[counts>1]):
            print("\r%s/%s"%(i+1, tot), end="")

            smooth_val = img_smooth_structured[img_structured==val]  # [(5, 3, 4), (5, 5, 6)]
            order_val = np.argsort(smooth_val, axis=None, order=("H", "S", "V"))

            partial_order = (img_structured[np.unravel_index(order, img_structured.shape[:2])]==val).flatten()  # Bool array with True when 'order' corresponds to 'val' 
            order[partial_order] = order[partial_order][order_val] # [4, 5]

    order = np.unravel_index(order, img_structured.shape[:2])
    
    print()
    return order


def create_order_single_channel(img, img_smooth=None, complete=False):
    """
    Creates an order on the pixels of img.
    The order is a tuple of two arrays of length img.shape[:2].product(),
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    The ordering is done by increasing value of the channel, and for
    same valued pixels, by the mean value of their neighbours.
    
    inputs:
    img: np.array. A 1-channel image, typically a greyscale image obtained with cv.cvtColor(X, cv.COLOR_BGR2GRAY)
    
    img_smoothed: np.array. Optionnal. A smoothed version of img used for ranking pixels based on their neighbourhood
    Must have the same dimension as img (typically using cv.GaussianBlur(img, (5, 5), 0)). Required if complete=True
    
    complete: bool. Optionnal. If complete=False, then the order will be arbitrary for pixels with same values
    If complete=True, the order will rank pixels with same values based on the mean value of their neighbours.
    
    ouput:
    order: (np.array, np.array). An array img.shape[:2].product().
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    """
    
    img_structured = img.copy()
    """
    [[3, 0], [2, 3], [8, 9], [5, 2]]
    """

    n_cols = img_structured.shape[1]
    order = np.argsort(img_structured, axis=None)
    """
    [1 7 2 0 3 6 4 5]
    """
    if complete:
        assert img_smooth is not None, "For a complete ordering, there must be an img_smoothed given"
        img_smooth_structured = img_smooth.copy()
        # Finding values that cannot be completely ordered with HSV
        values, counts = np.unique(img_structured, return_counts=True)
        """
        values = [0, 2, 3, 5, 8, 9]
        count = [1, 2, 2, 1, 1, 1]

        """
        tot = np.sum(counts>1)
        for i, val in enumerate(values[counts>1]):
            print("\r%s/%s"%(i+1, tot), end="")

            smooth_val = img_smooth_structured[img_structured==val]
            order_val = np.argsort(smooth_val, axis=None)

            partial_order = (img_structured[np.unravel_index(order, img_structured.shape[:2])]==val).flatten()  # Bool array with True when 'order' corresponds to 'val' 
            order[partial_order] = order[partial_order][order_val]

    order = np.unravel_index(order, img_structured.shape[:2])
    
    print()
    return order