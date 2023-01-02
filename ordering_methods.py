import numpy as np
import pandas as pd
import cv2
import warnings


def create_order_3_channels(img, img_smooth=None):
    """
    Creates an order on the pixels of img based on their 3 channels.
    The order is a tuple of two arrays of length img.shape[:2].product(),
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    The pixels are ordred by increasing 1st channel, then 2nd, then 3rd,
    then by increasing 1st, 2nd and 3rd channel of their neighbours.
    
    inputs:
    img: np.array. A 3-channel image, typically a HSV or BGR image obtained with cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
    
    img_smooth: Optionnal. np.array. A smoothed version of img used for ranking pixels based on their neighbourhood
    Must have the same dimension as img (typically using cv2.GaussianBlur(img, (5, 5), 0)).
    
    ouput:
    order: (np.array, np.array). An array img.shape[:2].product().
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    """

    indices = np.indices(img.shape[:2])

    if img_smooth is None:
        data = np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                                np.ravel(img[:, :, 0]), np.ravel(img[:, :, 1]), np.ravel(img[:, :, 2])))

        df = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3"], data=data)
        df = df.sort_values(by=["ch1", "ch2", "ch3"])

    else:
        data = np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                                np.ravel(img[:, :, 0]), np.ravel(img[:, :, 1]), np.ravel(img[:, :, 2]),
                                np.ravel(img_smooth[:, :, 0]), np.ravel(img_smooth[:, :, 1]),
                                np.ravel(img_smooth[:, :, 2])))

        df = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3", "smooth1", "smooth2", "smooth3"], data=data)
        df = df.sort_values(by=["ch1", "ch2", "ch3", "smooth1", "smooth2", "smooth3"])

    return df["row"].to_numpy(dtype=int), df["col"].to_numpy(dtype=int)


def create_order_single_channel(img, img_smooth=None):
    """
    Creates an order on the pixels of img.
    The order is a tuple of two arrays of length img.shape[:2].product(),
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    The ordering is done by increasing value of the channel, and for
    same valued pixels, by the mean value of their neighbours.
    
    inputs:
    img: np.array. A 1-channel image, typically a greyscale image obtained with cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    
    img_smoothed: np.array. Optionnal. A smoothed version of img used for ranking pixels based on their neighbourhood
    Must have the same dimension as img (typically using cv2.GaussianBlur(img, (5, 5), 0)).
       
    ouput:
    order: (np.array, np.array). An array img.shape[:2].product().
    (order[0][i], order[1][i]) is the coordinates of the i-th pixel.
    """

    indices = np.indices(img.shape)

    if img_smooth is None:
        data = np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                                np.ravel(img)))

        df = pd.DataFrame(columns=["row", "col", "ch1"], data=data)
        df = df.sort_values(by=["ch1"])

    else:
        data = np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                                np.ravel(img), np.ravel(img_smooth)))

        df = pd.DataFrame(columns=["row", "col", "ch1", "smooth1"], data=data)
        df = df.sort_values(by=["ch1", "smooth1"])

    return df["row"].to_numpy(dtype=int), df["col"].to_numpy(dtype=int)


def transfer_independent_channels(img, img_ref):
    """
    Transfer each histogram of a multichannel image independently,
    then reconstruct a mutlichannel image from each transfered channel
    """
    channels = cv2.split(img)
    ref_channels = cv2.split(img_ref)

    img_smooth = cv2.GaussianBlur(img, (5, 5), 0)

    order_1 = create_order_single_channel(channels[0], img_smooth[:, :, 0])
    order_2 = create_order_single_channel(channels[1], img_smooth[:, :, 1])
    order_3 = create_order_single_channel(channels[2], img_smooth[:, :, 2])

    order_ref_1 = create_order_single_channel(ref_channels[0])
    order_ref_2 = create_order_single_channel(ref_channels[1])
    order_ref_3 = create_order_single_channel(ref_channels[2])

    channels[0][order_1] = ref_channels[0][order_ref_1]
    channels[1][order_2] = ref_channels[1][order_ref_2]
    channels[2][order_3] = ref_channels[2][order_ref_3]

    return cv2.merge([channels[0], channels[1], channels[2]])


# TODO Sort with a smooth ref too

def transfer_with_dependence_euclidian(img, img_smooth, img_ref, seed=42):
    """
    Transfer the multidimension histogramm from one image to another.
    For each pixel of the reference image (chosen at random),
    finds the nearest pixel in the other image in term of Euclidian distance
    """
    assert img.shape == img_ref.shape, "This method is implemented for images with the same number of pixels"
    if img.dtype == np.dtype('uint8'):
        warnings.warn("img dtype is unsigned integer. Casting to short integer to compute euclidian distance")
        img = img.astype(dtype=np.short)

    if img_ref.dtype == np.dtype('uint8'):
        warnings.warn("img_ref dtype is unsigned integer. Casting to short integer to compute euclidian distance")
        img_ref = img_ref.astype(dtype=np.short)

    if img_smooth.dtype == np.dtype('uint8'):
        warnings.warn("img_smooth dtype is unsigned integer. Casting to short integer to compute euclidian distance")
        img_smooth = img_smooth.astype(dtype=np.short)

    indices = np.indices(img.shape[:2])
    data = np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                            np.ravel(img[:, :, 0]), np.ravel(img[:, :, 1]), np.ravel(img[:, :, 2]),
                            np.ravel(img_smooth[:, :, 0]), np.ravel(img_smooth[:, :, 1]),
                            np.ravel(img_smooth[:, :, 2])))

    df = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3", "smooth1", "smooth2", "smooth3"], data=data)

    indices_ref = np.indices(img_ref.shape[:2])

    data_ref = np.column_stack((np.ravel(indices_ref[0]), np.ravel(indices_ref[1]),
                                np.ravel(img_ref[:, :, 0]), np.ravel(img_ref[:, :, 1]), np.ravel(img_ref[:, :, 2])))

    df_ref = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3"], data=data_ref)

    img_transferred = img.copy()

    rng = np.random.default_rng(seed)
    random_index = rng.permutation(df_ref.index)

    for k, ind in enumerate(random_index):
        print("\r%s / %s    " % (k, len(df_ref)), end="")

        df["d_p"] = np.sqrt((df["ch1"] - df_ref.loc[ind, "ch1"]) ** 2 +
                            (df["ch2"] - df_ref.loc[ind, "ch2"]) ** 2 +
                            (df["ch3"] - df_ref.loc[ind, "ch3"]) ** 2)

        df["d_smooth_p"] = np.sqrt((df["smooth1"] - df_ref.loc[ind, "ch1"]) ** 2 +
                                   (df["smooth2"] - df_ref.loc[ind, "ch2"]) ** 2 +
                                   (df["smooth3"] - df_ref.loc[ind, "ch3"]) ** 2)

        df = df.sort_values(by=["d_p", "d_smooth_p"])

        img_transferred[df.loc[df.index[0], "row"], df.loc[df.index[0], "col"], :] = df_ref.loc[
            ind, ["ch1", "ch2", "ch3"]].to_numpy(
            dtype=int)

        df.drop(df.index[0], axis=0, inplace=True)

    return img_transferred


def compute_square_euclidian_distance_matrix(img, img_ref):
    """
    Compute a matrix of squared distances between the pixel of an image (R**2 + G**2 + B**2)
    img is an array of 3 channel pixels. img_ref is an array of 3 channel pixels
    if img.shape = (N, 3) and img_ref.shape = (M, 3), then return a (M, N) matrix
    Columns correspond to the flattened position of a pixel of img,
    and rows to the flatten position of a pixel of img_ref
    """
    assert img.shape[-1] == 3 & img_ref.shape[-1] == 3, "img and img_ref should have 3 channels"
    if img.dtype == np.dtype('uint8'):
        warnings.warn("img dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img = img.astype(dtype=np.int32)

    if img_ref.dtype == np.dtype('uint8'):
        warnings.warn("img_ref dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img_ref = img_ref.astype(dtype=np.int32)

    ch1_img, ch1_ref = np.meshgrid(img[:, 0].flatten(), img_ref[:, 0].flatten(), sparse=True)
    ch2_img, ch2_ref = np.meshgrid(img[:, 1].flatten(), img_ref[:, 1].flatten(), sparse=True)
    ch3_img, ch3_ref = np.meshgrid(img[:, 2].flatten(), img_ref[:, 2].flatten(), sparse=True)

    dist_img = np.square(ch1_img - ch1_ref) + np.square(ch2_img - ch2_ref) + np.square(ch3_img - ch3_ref)

    return dist_img


def transfer_with_dependence_euclidian_heuristic(img, img_smooth, img_ref, batch_size=100, seed=42):
    assert img.shape == img_ref.shape, "Method only works for same shape images." \
                                 "Resample them so they have the same shape"

    if img.dtype == np.dtype('uint8'):
        warnings.warn("img dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img = img.astype(dtype=np.int32)

    if img_ref.dtype == np.dtype('uint8'):
        warnings.warn("img_ref dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img_ref = img_ref.astype(dtype=np.int32)

    if img_smooth.dtype == np.dtype('uint8'):
        warnings.warn("img dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img_smooth = img_smooth.astype(dtype=np.int32)

    n_pixels = np.product(img_ref.shape[:2])
    indices = pd.DataFrame(index=range(n_pixels), columns=["ind"], data=range(n_pixels))

    # TODO doing the match not at random but by crescent distance from the sides of the RGB cube
    # Choosing pixels from img_ref at random
    rng = np.random.default_rng(seed)
    indices_ref = pd.DataFrame(index=range(n_pixels), columns=["ind"], data=rng.permutation(np.arange(n_pixels)))

    # full_order: index = pixel in img, column = pixel in img_ref
    full_order = pd.DataFrame(index=range(n_pixels), columns=["ind"], data=np.zeros(n_pixels))

    for k in range(n_pixels // batch_size + 1):
        print("\r%s / %s    " % (k+1, n_pixels // batch_size+1), end="")
        """
        Example:
        indices = 
            ind
        0    4    
        1    5
        2    9
        3   12
        
        indices_ref = 
            ind
        0    7
        1    3
        2    0
        """
        # Splitting the random order of pixels in different batch for faster processing
        i_min, i_max = k * batch_size, min((k + 1) * batch_size, len(indices_ref))
        if i_min == i_max:
            i_max += 1
        """
        i_min, i_max = 0, 3
        """
        indices_ref_batch = indices_ref.loc[i_min:(i_max-1), "ind"].to_numpy()
        """[7, 3, 0]"""

        while indices_ref_batch.size > 0:
            rows, cols = np.unravel_index(indices["ind"], img.shape[:2])
            """[a, b, c, d], [A, B, C, D]"""
            rows_ref, cols_ref = np.unravel_index(indices_ref_batch, img_ref.shape[:2])
            """[a', b', c'], [A', B', C']"""
            # Computing distance matrices
            dist_img = compute_square_euclidian_distance_matrix(img[rows, cols, :], img_ref[rows_ref, cols_ref, :])
            """
            [[1, 0, 0, 2],
             [5, 7, 9, 8],
             [3, 1, 1, 2]]
            """
            dist_smooth_img = compute_square_euclidian_distance_matrix(img_smooth[rows, cols, :],
                                                                       img_ref[rows_ref, cols_ref, :])
            """
            [[4, 2, 1, 3],
             [6, 6, 6, 6],
             [9, 3, 0, 0]]
            """

            # Finding the minimal distances for each row
            pixel_min_val = dist_img.min(axis=1)  # Min distance retained
            """[0, 5, 1]"""
            dist_min = np.tile(pixel_min_val, (dist_img.shape[1], 1)).T
            """
            [[0, 0, 0, 0],
             [5, 5, 5, 5],
             [1, 1, 1, 1]]
            """

            dist_min = np.isclose(dist_img, dist_min)
            """
            [[False, True, True, False],
             [True, False, False, False],
             [False, True, True, False]]
            """

            # For distances that are equal, sorting by neighbour
            dist_smooth_img[~dist_min] = 200000  # More than dist_max=3*255**2
            """
            [[2e5, 2, 1, 2e5],
             [6, 2e5, 2e5, 2e5],
             [2e5, 3, 0, 2e5]]
            """
            pixel_min = np.argmin(dist_smooth_img, axis=1)  # array of shape (i_min - i_max,)
            """[2, 0, 2]"""

            unique, unique_counts = np.unique(pixel_min, return_counts=True)
            """[0, 2], [1, 2]"""
            is_unique = np.isin(pixel_min, unique[unique_counts == 1])
            """[False, True, False]"""

            for pixel in unique[unique_counts > 1]:
                pixel_min_val_temp = pixel_min_val
                """[0, 5, 1]"""
                pixel_min_val_temp[pixel_min != pixel] = 200000
                """[0, 2e5, 1]"""
                pixel_ref = np.argmin(pixel_min_val_temp)
                """0"""
                is_unique[pixel_ref] = True
                """[True, True, False]"""

            # full_order: index = pixel in img, column = pixel in img_ref
            full_order.loc[indices.loc[pixel_min[is_unique], "ind"], "ind"] = indices_ref_batch[is_unique]
            """full_order.loc[[9, 4], "ind"] = [7, 3]"""

            matches = np.isin(indices.index, pixel_min[is_unique])  # TODO The is_unique should be redundant
            """[True, False, True, False]"""
            # Removing the pixels from img that have been chosen to the list of available pixels
            indices = indices[~matches]
            indices.reset_index(drop=True, inplace=True)

            # Removing the pixels from the batch of img_ref that have been assigned a match
            indices_ref_batch = indices_ref_batch[~is_unique]

    img_copy = img.copy()

    rows, cols = np.unravel_index(full_order.index, img.shape[:2])
    rows_ref, cols_ref = np.unravel_index(full_order["ind"].to_numpy(dtype=int), img_ref.shape[:2])

    img_copy[rows, cols, :] = img_ref[rows_ref, cols_ref, :]

    return img_copy


def transfer_based_on_edges(img,  img_ref, kernel_size=5):
    assert img.shape == img_ref.shape, "Method only works for same shape images." \
                                 "Resample them so they have the same shape"

    img_smooth = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img_ref_smooth = cv2.GaussianBlur(img_ref, (kernel_size, kernel_size), 0)

    if img.dtype == np.dtype('uint8'):
        warnings.warn("img dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img = img.astype(dtype=np.int32)
        img_smooth = img_smooth.astype(dtype=np.int32)

    if img_ref.dtype == np.dtype('uint8'):
        warnings.warn("img_ref dtype is unsigned integer. Casting to int32 to compute euclidian distance")
        img_ref = img_ref.astype(dtype=np.int32)
        img_ref_smooth = img_ref_smooth.astype(dtype=np.int32)

    edges_pivot_points = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                          [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255],
                          [127, 0, 0], [127, 255, 0], [127, 0, 255], [127, 255, 255],
                          [0, 127, 0], [255, 127, 0], [0, 127, 255], [255, 127, 255],
                          [0, 0, 127], [255, 0, 127], [0, 255, 127], [255, 255, 127]]

    df_edges = pd.DataFrame(columns=["E"+str(i) for i in range(len(edges_pivot_points))])
    df_ref_edges = pd.DataFrame(columns=["E"+str(i) for i in range(len(edges_pivot_points))])

    for i, p in enumerate(edges_pivot_points):
        df = pd.DataFrame()
        df_ref = pd.DataFrame()
        df["D_img"] = np.square(img.reshape((-1, 3)) - np.array(p)).sum(axis=1)
        df["D_smooth"] = np.square(img_smooth.reshape((-1, 3)) - np.array(p)).sum(axis=1)
        df_ref["D_img"] = np.square(img_ref.reshape((-1, 3)) - np.array(p)).sum(axis=1)
        df_ref["D_smooth"] = np.square(img_ref_smooth.reshape((-1, 3)) - np.array(p)).sum(axis=1)

        df = df.sort_values(by=["D_img", "D_smooth"])
        df_ref = df_ref.sort_values(by=["D_img", "D_smooth"])

        df_edges["E" + str(i)] = df.index
        df_ref_edges["E" + str(i)] = df_ref.index

    """
        E1  E2  E3
    0    1   3   0
    1    0   1   3
    2    3   0   4
    3    2   2   1
    4    4   4   2
    
    order_with_duplicates:
        E1  E2  E3
    0  True  True  True
    1  False False False
    2  False False True
    3  True  False False
    4  False False False
    """

    order_with_duplicates = pd.Series(df_edges.to_numpy().flatten()).duplicated().to_numpy().reshape(df_edges.shape)
    order_ref_with_duplicates = pd.Series(df_ref_edges.to_numpy().flatten()).duplicated().to_numpy().reshape(df_ref_edges.shape)

    n_pixels = img.shape[0]*img.shape[1]

    # TODO: is this ordering has a lot of biases ?
    for i in range(len(edges_pivot_points)):
        nan_tail = np.array((n_pixels - order_with_duplicates[:, i].sum())*[n_pixels])
        df_edges["E" + str(i)] = np.append(df_edges["E" + str(i)][order_with_duplicates[:, i]].to_numpy(), nan_tail)

        nan_tail = np.array((n_pixels - order_ref_with_duplicates[:, i].sum())*[n_pixels])
        df_ref_edges["E" + str(i)] = np.append(df_ref_edges["E" + str(i)][order_ref_with_duplicates[:, i]].to_numpy(), nan_tail)
    """
        E1  E2  E3
    0    1   3   0
    1    2   5   4
    2    5   5   5
    3    5   5   5
    4    5   5   5
    """
    order = df_edges.to_numpy().ravel()
    order = np.unravel_index(order[order != n_pixels], img.shape[:2])

    order_ref = df_ref_edges.to_numpy().ravel()
    order_ref = np.unravel_index(order_ref[order_ref != n_pixels], img_ref.shape[:2])

    img_copy = img.copy()
    img_copy[order] = img_ref[order_ref]

    return img_copy

