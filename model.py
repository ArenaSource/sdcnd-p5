import os
import cv2
import glob
import random
import math
import collections
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from scipy.ndimage.measurements import label

np.random.seed(1234)


class Classifier(object):
    def __init__(self, path='./datasets', limit=None):
        self._positive = [file for file in glob.iglob(path + '/vehicles/**/*.png', recursive=True)]
        self._negative = [file for file in glob.iglob(path + '/non-vehicles/**/*.png', recursive=True)]
        # Shuffle
        np.random.shuffle(self._positive)
        np.random.shuffle(self._negative)
        # Balance data set
        size = int(limit / 2) if limit else min((len(self._positive), len(self._negative)))
        self._positive = self._positive[:size]
        self._negative = self._negative[:size]
        # Features
        X_raw = [Image(cv2.imread(file)).features for file in self._positive + self._negative]
        self.y = [1] * len(self._positive) + [0] * len(self._negative)
        # Normalize
        self.scaler = StandardScaler().fit(X_raw)
        self.X = self.scaler.transform(X_raw)
        # Train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        self.model = LinearSVC().fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)

    def image(self, vehicle, offset=0):
        filename = self._positive[offset] if vehicle else self._negative[offset]
        return Image(cv2.imread(filename))

    def predict(self, X):
        X = self.scaler.transform(X.reshape(1, -1))
        return self.model.predict(X)[0]


class Image(object):
    def __init__(self, pixels, metadata=None):
        self.pixels = pixels
        self.metadata = metadata

    @classmethod
    def imread(cls, filename):
        return Image(cv2.imread(filename))

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def RGB(self):
        return self.copy(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB))

    @property
    def HLS(self):
        return self.copy(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2HLS))

    @property
    def YCrCb(self):
        return self.copy(cv2.cvtColor(self.pixels, cv2.COLOR_RGB2YCrCb))

    @property
    def Y(self):
        return self.copy(self.YCrCb.pixels[:, :, 0])

    @property
    def Cr(self):
        return self.copy(self.YCrCb.pixels[:, :, 1])

    @property
    def Cb(self):
        return self.copy(self.YCrCb.pixels[:, :, 2])

    @property
    def features(self):
        return np.concatenate((self.Y.hog(), self.Cr.hog(), self.Cb.hog(), self.histogram(), self.spatial()))

    @property
    def bbox(self):
        return [[0, 400], [self.shape[1], 600]]

    def windows(self, layers=5, scale_factor=0.2, overlap=0.5):
        max_box_size = self.bbox[1][1] - self.bbox[0][1]
        min_box_size = max_box_size * scale_factor
        size_diff = (max_box_size - min_box_size) / (layers - 1)
        windows = []

        for i in range(layers):
            width = self.bbox[1][0] - self.bbox[0][0]
            size = int(max_box_size - size_diff * i)
            n_windows = math.floor(width / (size * (1.0 - overlap)))
            step = size + ((width - (size * n_windows)) / n_windows)
            step = (width / n_windows)
            step += (step - size) / n_windows # Adjust to rigth
            layer = []
            for j in range(n_windows):
                box = self.bbox
                box[0][0] = int(box[0][0] + j * step)
                box[1][0] = box[0][0] + size
                box[1][1] = box[0][1] + size
                image = Image(pixels=cv2.resize(self.pixels[box[0][1]:box[1][1], box[0][0]:box[1][0]], (64, 64)),
                              metadata={'box': box})
                layer.append(image)
            windows.append(layer)

        return windows

    def search(self, classifier, layers=5, scale_factor=0.2, overlap=0.5):
        matches = []
        for w in sum(self.windows(layers, scale_factor, overlap), []):
            if classifier.predict(w.features) == 1:
                matches.append(w.metadata['box'])
        return matches

    def heatmap(self, results, threshold=1):
        pixels = np.zeros_like(self.pixels[:, :, 0]).astype(np.float)
        for box in results:
            pixels[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        pixels[pixels <= threshold] = 0
        return Image(pixels=np.clip(pixels, 0, 255))

    def labels(self):
        labeled, cars = label(self.pixels)
        labels = []
        for idx in range(1, cars + 1):
            nonzero = (labeled == idx).nonzero()
            y = np.array(nonzero[0])
            x = np.array(nonzero[1])
            labels.append(((np.min(x), np.min(y)), (np.max(x), np.max(y))))
        return labels

    def copy(self, pixels=None):
        if pixels is None:
            pixels = np.copy(self.pixels)
        return Image(pixels)

    def __and__(self, image):
        return self.copy(self.pixels & image.pixels)

    def __or__(self, image):
        return self.copy(self.pixels | image.pixels)

    def histogram(self, bins=32):
        h = []
        for i in range(self.pixels.shape[-1]):
            h.append(np.histogram(self.pixels[:, :, i], bins=bins, range=(0, 256))[0])
        return np.concatenate(h)

    def spatial(self, size=16):
        return cv2.resize(self.pixels, (size, size)).ravel()

    def hog(self, bins=11, ppc=8, cpb=2, debug=False):
        assert len(self.shape) == 2, "Multichannel operation is not supported"
        output = hog(self.pixels, orientations=bins, pixels_per_cell=(ppc, ppc),
                     cells_per_block=(cpb, cpb), transform_sqrt=False,
                     visualise=debug, feature_vector=True, block_norm='L2-Hys')
        if debug:
            return output[0], Image(output[1])
        return output

    def draw_box(self, box, color=(0, 0, 255), thick=2):
        cv2.rectangle(self.pixels, tuple(box[0]), tuple(box[1]), color, thick)
        return self

    def show(self, cmap=None):
        if len(self.shape) == 3:
            plt.imshow(self.RGB.pixels, cmap=cmap)
        else:
            plt.imshow(self.pixels, cmap='gray')


class Tracker(object):

    def __init__(self, classifier, layers=5, scale_factor=0.2, overlap=0.5, qsize=20, threshold=20):
        self.classifier = classifier
        self.layers = layers
        self.scale_factor = scale_factor
        self.overlap = overlap
        self.threshold = threshold
        self.image = None
        self._results = collections.deque([], qsize)

    @property
    def results(self):
        return sum(self._results, [])

    @property
    def output(self):
        image = self.image.copy()
        for box in self.image.heatmap(self.results, threshold=self.threshold).labels():
            image.draw_box(box)
        return image

    def run(self, pixels):
        self.image = Image(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        self._results.append(
            self.image.search(classifier=self.classifier, layers=self.layers,
                              scale_factor=self.scale_factor, overlap=self.overlap)
        )
