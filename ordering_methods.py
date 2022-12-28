import numpy as np
import pandas as pd
import cv2


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


def convert_independent_channels(img, img_ref):
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
    assert img.shape == img_ref.shape, "This method is implemented for images with the same number of pixels"

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

        img_transferred[df.loc[df.index[0], "row"], df.loc[df.index[0], "col"], :] = df_ref.loc[ind, ["ch1", "ch2", "ch3"]].to_numpy(
            dtype=int)

        df.drop(df.index[0], axis=0, inplace=True)

    return img_transferred


def bad_transfer_with_dependence_euclidian(img, img_smooth, img_ref):
    assert img.shape == img_ref.shape, "This method is implemented for images with the same number of pixels"

    indices = np.indices(img.shape[:2])

    data = np.column_stack((np.ravel(indices[0]), np.ravel(indices[1]),
                            np.ravel(img[:, :, 0]), np.ravel(img[:, :, 1]), np.ravel(img[:, :, 2]),
                            np.ravel(img_smooth[:, :, 0]), np.ravel(img_smooth[:, :, 1]),
                            np.ravel(img_smooth[:, :, 2])))

    df = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3", "smooth1", "smooth2", "smooth3"], data=data)
    df["d_0"] = np.sqrt(df["ch1"]**2 + df["ch2"]**2 + df["ch3"]**2)
    df["d_ch1"] = np.sqrt((df["ch1"]-255)**2 + df["ch2"]**2 + df["ch3"]**2)
    df["d_ch2"] = np.sqrt(df["ch1"]**2 + (df["ch2"]-255)**2 + df["ch3"]**2)

    df["d_smooth0"] = np.sqrt(df["smooth1"] ** 2 + df["smooth2"] ** 2 + df["smooth3"] ** 2)
    df["d_smooth1"] = np.sqrt((df["smooth1"]-255) ** 2 + df["smooth2"] ** 2 + df["smooth3"] ** 2)
    df["d_smooth2"] = np.sqrt(df["smooth1"] ** 2 + (df["smooth2"] - 255) ** 2 + df["smooth3"] ** 2)

    df = df.sort_values(by=["d_0", "d_ch1", "d_ch2", "d_smooth0", "d_smooth1", "d_smooth2"])

    indices_ref = np.indices(img_ref.shape[:2])

    data_ref = np.column_stack((np.ravel(indices_ref[0]), np.ravel(indices_ref[1]),
                                np.ravel(img_ref[:, :, 0]), np.ravel(img_ref[:, :, 1]), np.ravel(img_ref[:, :, 2])))

    df_ref = pd.DataFrame(columns=["row", "col", "ch1", "ch2", "ch3"], data=data_ref)
    df_ref["d_0"] = np.sqrt(df_ref["ch1"]**2 + df_ref["ch2"]**2 + df_ref["ch3"]**2)
    df_ref["d_ch1"] = np.sqrt((df_ref["ch1"]-255)**2 + df_ref["ch2"]**2 + df_ref["ch3"]**2)
    df_ref["d_ch2"] = np.sqrt(df_ref["ch1"]**2 + (df_ref["ch2"]-255)**2 + df_ref["ch3"]**2)

    df_ref = df_ref.sort_values(by=["d_0", "d_ch1", "d_ch2"])

    img_transferred = img.copy()
    img_transferred[df["row"].to_numpy(dtype=int), df["col"].to_numpy(dtype=int), :] = df_ref[df_ref["row"].to_numpy(dtype=int), df_ref["col"].to_numpy(dtype=int), :]
    return img_transferred
