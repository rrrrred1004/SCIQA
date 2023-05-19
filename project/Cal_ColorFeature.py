import os
import cv2
import colorsys
import numpy as np
import pandas as pd
from PIL import Image

def create_hls_array(image):
    """
    Creates a numpy array holding the hue, lightness
    and saturation values for the Pillow image.
    """

    # pixels = image.load()

    hls_array = np.empty(shape=(image.shape[0], image.shape[1], 3), dtype=float)

    for row in range(0, image.shape[0]):
        for column in range(0, image.shape[1]):
            # rgb = pixels[column, row]
            rgb = image[row, column]
            hls = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

            hls_array[row, column, 0] = hls[0]
            hls_array[row, column, 1] = hls[1]
            hls_array[row, column, 2] = hls[2]

    return hls_array

def cal_saturation(hls_array):
    h = hls_array[:, :, 0]
    s = hls_array[:, :, 2]
    s_array = s[np.nonzero(h)]

    if s_array.size != 0:

        # 求均值
        mean = np.mean(s_array)
        # 求中位数
        median = np.median(s_array)
        # 求25%分位数：
        p25 = np.percentile(s_array, 25)
        # 求75%分位数：
        p75 = np.percentile(s_array, 75)

        list = [mean, median, p25, p75]

    else:
        list = [0, 0, 0, 0]

    list = np.array(list)
    list = list.reshape(1, -1)
    # print(fs.shape)
    return list

print("| RGB to HSL Conversion |")
path = 'D:/Projects/Database/SCIdatabase/Distorted images/'
file_list = os.listdir(path)
# file_list.sort(key=lambda x: int((x.split('.')[0])[3:]) if x.endswith('bmp') else 0)

fc = np.empty(shape=[0, 4])

for file in file_list:
    img_path = os.path.join(path, file)
    img = cv2.imread(img_path)
    img_test = img.copy()

    # scale 1
    print('Start multiscale processing of', file)
    print('scale 1')
    hls_array = create_hls_array(img_test)
    result = cal_saturation(hls_array)
    fc = np.row_stack((fc, result))

    # for i in np.arange(4):
    #     img_test = cv2.pyrDown(img_test)
    #     print('scale', i+2)
    #     hls_array = create_hls_array(img_test)
    #     result = cal_saturation(hls_array)
    #     fs = np.row_stack((fs, result))

# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
#
# print(fc)
# print(fc.shape)
np.save('fc.npy', fc)