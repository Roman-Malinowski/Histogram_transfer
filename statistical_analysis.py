import os.path as op
import os
import numpy as np
import cv2 as cv
import pandas as pd



def cvt_ruderman(img_XYZ):

    XYZ2LMS = np.array([[0.3897, 0.6890, -0.0787], [-0.2298, 1.1834, 0.0464], [0, 0, 1]])
    
    img_LMS = np.matmul(img_XYZ, XYZ2LMS.T)
    log_img = np.log10(img_LMS)
    
    A = np.diag([np.sqrt(3)/3, np.sqrt(6)/6, np.sqrt(2)/2])
    B = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
    LMS2ruderman = np.matmul(B.T, A.T)
    
    log_ruderman = np.matmul(log_img, LMS2ruderman)
    
    return log_img, log_ruderman


def cvt_back_ruderman(log_ruderman):
    
    A = np.diag([np.sqrt(3)/3, np.sqrt(6)/6, np.sqrt(2)/2])
    B = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
    ruderman2LMS = np.matmul(A, B)
    
    log_img = np.matmul(log_ruderman, ruderman2LMS)
    lms = 10**log_img
    XYZ2LMS = np.array([[0.3897, 0.6890, -0.0787], [-0.2298, 1.1834, 0.0464], [0, 0, 1]])
    LMS2XYZ = np.linalg.inv(XYZ2LMS.T)
    
    img_XYZ = np.matmul(lms, LMS2XYZ)

    return lms, img_XYZ


def compare_covariances(list_paths: list) -> pd.DataFrame:
    """
    Compute a dataframe storing every covariance between channels of different color spaces from a list of images paths.
    """

    df_cov = pd.DataFrame(columns=["image_path_1", "Color space", "cov_12", "cov_13", "cov_23"])
    
    # BGR, LAB, HSV, YCrCb, LUV, XYZ, LMS, Ruderman
    
    for path in list_paths:

        image_BGR = cv.imread(path)
        image_LAB = cv.cvtColor(image_BGR, cv.COLOR_BGR2LAB)
        image_HSV = cv.cvtColor(image_BGR, cv.COLOR_BGR2HSV)
        image_YCrCb = cv.cvtColor(image_BGR, cv.COLOR_BGR2YCrCb)
        image_LUV = cv.cvtColor(image_BGR, cv.COLOR_BGR2Luv)
        image_XYZ = cv.cvtColor(image_BGR, cv.COLOR_BGR2XYZ)
        image_LMS, image_ruderman = cvt_ruderman(image_XYZ)

        dict_image = dict(zip(["BGR", "LAB", "HSV", "YCrCb", "LUV", "XYZ", "LMS", "RUDERMAN"], [image_BGR, image_LAB, image_HSV, image_YCrCb, image_LUV, image_XYZ, image_LMS, image_ruderman]))

        for color_space, img in dict_image.items():
            channel_1, channel_2, channel_3 = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
            mask = (channel_1!=-np.inf) & (channel_2!=-np.inf) & (channel_3!=-np.inf)
            cov = np.cov([channel_1[mask], channel_2[mask], channel_3[mask]])
            df_cov.loc[len(df_cov), :] = [path, color_space, cov[0,1], cov[0,2], cov[1,2]]
    
    return df_cov


def explore_data_folder(root, list_extensions):
    list_paths = []
    
    leafs = [op.join(root, leaf) for leaf in os.listdir(root)]
    leafs_folders = [leaf for leaf in leafs if op.isdir(leaf)]
    
    for leaf in leafs_folders:
        list_paths += explore_data_folder(leaf, list_extensions)

    list_paths += [leaf for leaf in leafs if op.splitext(leaf)[-1] in list_extensions]
    
    return list_paths


if __name__=="__main__":
    # prefix = "/work/scratch/malinoro/histogram_transfer/data"
    prefix = "/Users/roman/Code/Histogram_transfer/data/"

    sources = [source for source in os.listdir(prefix) if op.isdir(op.join(prefix, source))]
    list_extensions = [".ppm", ".tiff", ".jpg"]

    list_paths = explore_data_folder(prefix, list_extensions)
    print(list_paths)
    
    df = compare_covariances(list_paths)
    print(df)

