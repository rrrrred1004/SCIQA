import os
import cv2
import numpy as np
import pandas as pd
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
import FeatureExtract
import warnings
warnings.filterwarnings("ignore")


# main
array = np.empty(shape=[0, 64])
train_path = './SIQAD/references/'
file_list = os.listdir(train_path)
# file_list.sort(key=lambda x: int((x.split('.')[0])[3:]) if x.endswith('bmp') else 0)
for file in file_list:
    print(file)
    img_path = os.path.join(train_path, file)
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    face = img / 255.0
    patch_size = (8, 8)
    data = extract_patches_2d(face, patch_size, max_patches=1000)
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data = data - intercept
    data /= np.std(data, axis=0)
    array = np.append(array, data, axis=0)
print(array.shape)
print(np.isnan(array).any())
print('Learning the dictionary')
t0 = time()
patch_size = (8, 8)
dico = MiniBatchDictionaryLearning(n_components=128, alpha=1, n_iter=500)
V = dico.fit(array).components_
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure(figsize=(6.5, 4))
for i, comp in enumerate(V[:128]):
    plt.subplot(8, 16, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from %d patches' % (len(array)),
                 fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85)
plt.show()
plt.savefig('dic.png', dpi=300, bbox_inches='tight')
print(V.shape)


N = 513
test_input = np.empty(shape=[N, 0])
data_test = np.empty(shape=[0, 64])
code_all = np.empty(shape=[0, 128])
print('Start sparse representation')
test_path = './SIQAD/DistortedImages/'
test_list = os.listdir(test_path)
# test_list.sort(key=lambda x: int((x.split('.')[0])[3:]) if x.endswith('bmp') else 0)
for file in test_list:
    img_path = os.path.join(test_path, file)
    img = cv2.imread(img_path, 0)
    img_test = img.copy()

    # scale 1
    print('Start multiscale processing of', file)
    print('scale 1')
    face = img_test
    patch_size = (8, 8)
    data = extract_patches_2d(face, patch_size)
    data = data.reshape(data.shape[0], -1)
    data1 = data / 255.0
    intercept = np.mean(data1, axis=0)
    data_test = data1 - intercept

    # sparse representation
    dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
    code = dico.transform(data_test)
    code = np.array(code)

    # feature extraction
    f = np.empty(shape=[N, 0])
    f1 = FeatureExtract.extract_patch_energy(code, data)
    f2 = FeatureExtract.estimate_GGD_parameters(code)
    f3 = FeatureExtract.extract_log_normal_distribution_feature(code)
    f4 = FeatureExtract.atoms_count(code)
    f = np.append(f, f1)
    f = np.append(f, f2)
    f = np.append(f, f3)
    f = np.append(f, f4)
    f = f.reshape(-1, 1)
    test_input = np.append(test_input, f, axis=1)

    # multi-scale
    for i in np.arange(4):
        img_test = cv2.pyrDown(img_test)
        print('scale', i+2)
        # cv2.imshow('img_test', img_test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        face = img_test
        patch_size = (8, 8)
        data = extract_patches_2d(face, patch_size)
        data = data.reshape(data.shape[0], -1)
        data1 = data / 255.0
        intercept = np.mean(data1, axis=0)
        data_test = data1 - intercept

        # sparse representation
        dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
        code = dico.transform(data_test)
        code = np.array(code)

        # feature extraction
        f = np.empty(shape=[N, 0])
        f1 = FeatureExtract.extract_patch_energy(code, data)
        f2 = FeatureExtract.estimate_GGD_parameters(code)
        f3 = FeatureExtract.extract_log_normal_distribution_feature(code)
        f4 = FeatureExtract.atoms_count(code)
        f = np.append(f, f1)
        f = np.append(f, f2)
        f = np.append(f, f3)
        f = np.append(f, f4)
        f = f.reshape(-1,1)
        test_input = np.append(test_input, f, axis=1)
    print(test_input.shape)
    dt1 = time() - t0
    print('done in %.2fs.' % dt1)

test_input = np.transpose(test_input)
print(test_input.shape)
np.save('*.npy', test_input)

