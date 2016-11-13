# coding:utf-8
import numpy as np
from scipy import ndimage
from skimage.transform import rescale
from sklearn.feature_extraction.image import extract_patches_2d
from elm_estimator import ELMRegressor


def filter_img(img):
    filters = [
        np.array([[-1, 0, 1]]),
        np.array([[-1, 0, 1]]).T,
        np.array([[1, 0, -2, 0, 1]]),
        np.array([[1, 0, -2, 0, 1]]).T,
    ]

    h, w = img.shape
    res = np.zeros((h, w, 1 + len(filters)))
    res[:, :, 0] = img
    for i, f in enumerate(filters):
        res[:, :, i + 1] = ndimage.convolve(img, f, mode='constant')
    return res


def extract(imgs, n_patches, patch_size):
    n = len(imgs)
    n_patches_per_img = int(n_patches / n) + (1 if n_patches % n != 0 else 0)
    features = []
    patches = []
    for hr_img in imgs:
        mr_img = rescale(rescale(hr_img, 0.5), 2.0)
        diff = hr_img - mr_img
        f = extract_patches_2d(filter_img(mr_img), patch_size)
        p = extract_patches_2d(diff, patch_size)

        idx = np.random.randint(len(f), size=n_patches_per_img)
        features.append(f[idx])
        patches.append(p[idx])

    features = np.concatenate(features)
    patches = np.concatenate(patches)

    # resize
    m = features.shape[0]
    features = features.reshape(m, -1)[:n_patches]
    patches = patches.reshape(m, -1)[:n_patches]

    return features, patches


def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img


class ELMSuperResolution(object):

    def __init__(self, n_patches=100000, n_hidden=100,
                 patch_size=(5, 5), reg=None):
        self.n_patches = n_patches
        self.n_hidden = n_hidden
        self.patch_size = patch_size
        self.reg = reg

        if self.reg is None:
            self.reg = ELMRegressor(n_hidden=self.n_hidden)

    def fit(self, imgs):
        features, patches = extract(imgs[1:], self.n_patches, self.patch_size)
        self.reg.fit(features, patches)

    def upscale(self, lr_img):
        mr_img = rescale(lr_img, 2)
        hr_img = np.zeros(mr_img.shape)
        weight = np.zeros(mr_img.shape)
        h, w = mr_img.shape
        features = extract_patches_2d(filter_img(mr_img), self.patch_size)
        features = features.reshape(len(features), -1)
        pred = self.reg.predict(features)
        f_cnt = 0
        py, px = self.patch_size
        for y in range(h-py+1):
            for x in range(w-px+1):
                p = pred[f_cnt]
                hr_img[y:y+py, x:x+px] += p.reshape(self.patch_size)
                weight[y:y+py, x:x+px] += 1
                f_cnt += 1
        hr_img /= weight
        hr_img += mr_img
        hr_img = clip(hr_img)

        return hr_img
