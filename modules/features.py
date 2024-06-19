from skimage.feature import hog
from skimage import color
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


def flatten_image(image):
    return image.flatten()


def extract_hog_features(image):
    image_gray = color.rgb2gray(image)  # Convertir l'image en niveaux de gris
    features, hog_image = hog(image_gray, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
    return features, hog_image


def extract_sift_features(image):
    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


# Feature extraction methods
def extract_features(images, method='flatten'):
    if method == 'flatten':
        return np.array([flatten_image(image) for image in images])
    elif method == 'hog':
        features_list = []
        for image in images:
            features, _ = extract_hog_features(image)  # We ignore the hog_image here
            features_list.append(features)
        return np.array(features_list)
    elif method == 'sift':
        sift_features = []
        for image in images:
            _, descriptors = extract_sift_features(image)
            if descriptors is not None:
                # Flatten and pad/truncate descriptors to a fixed length
                descriptors = descriptors.flatten()
                if len(descriptors) >= 3072:
                    descriptors = descriptors[:3072]
                else:
                    descriptors = np.pad(descriptors, (0, 3072 - len(descriptors)))
                sift_features.append(descriptors)
            else:
                sift_features.append(np.zeros(3072))  # Pad with zeros if no descriptors
        return np.array(sift_features)


# Standardize features
def standardize_features(features):
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    return features
