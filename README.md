# CIFAR-10 Project
Abdelhamid MOUTI, Thomas FERMELI-FURIC


## Installation

The first step is to download the following 'data' repository and place it at the root of the project:

https://epitafr-my.sharepoint.com/:f:/g/personal/thomas_fermeli-furic_epita_fr/EkxSxWSsRLlEgGO_pFBQ8XABZIbrJn9dbiuUGVAYoW50bw?e=WP5LgG

The second step is to download the following 'models' repository and place it at the root of the project:

https://epitafr-my.sharepoint.com/:f:/g/personal/thomas_fermeli-furic_epita_fr/EnWtNMngl35Ev3v5QcSllhsBsqr_SJfzmLKlndh8VY8dlg?e=b9AImq

Then, you can install the project dependencies by running the following command at the root of the project:

```
pip install -r requirements.txt
```

## Configuration

In the file /modules/config.py, you will find the following global variables:
- DATA_DIR: the path to the 'data' repository
- MODELS_DIR: the path to the 'models' repository
- N_TRAIN_SAMPLES: the number of images to use for training
- N_TEST_SAMPLES: the number of images to use for testing

If the project architecture has been respected, you should not need to modify DATA_DIR and MODELS_DIR. However, you might want to modify the values of N_TRAIN_SAMPLES and N_TEST_SAMPLES to speed up the training and testing processes.

## Data loading

The following code can be used to load the CIFAR-10 dataset:

```
from modules.dataset import load_cifar10_dataset
from modules.config import DATA_DIR

# Load the CIFAR-10 dataset
train_data, train_labels, test_data, test_labels, label_names = load_cifar10_dataset(DATA_DIR)
```

## Features extraction and visualisation

The following code can be used to extract and visualize features from a sample image:

```
from modules.dataset import load_cifar10_dataset
from modules.features import flatten_image, extract_hog_features, extract_sift_features
from modules.plots import visualize_flattened_image, visualize_hog_features, visualize_sift_features

from modules.config import DATA_DIR

# Load the CIFAR-10 dataset
train_data, train_labels, test_data, test_labels, label_names = load_cifar10_dataset(DATA_DIR)

# Perform feature extraction and visualization on a sample image
image = train_data[0]

# Flatten
flattened_image = flatten_image(image)
visualize_flattened_image(image, flattened_image)

# HOG
hog_features, hog_image = extract_hog_features(image)
visualize_hog_features(image, hog_image)

# SIFT
keypoints, descriptors = extract_sift_features(image)
visualize_sift_features(image, keypoints)
```

## Training the models

When executed at the root of the project, the following code can be used to train the models:

```
python -m modules.modeling.train
```

This Python file will train the three models on each of the three feature extraction methods. The models will be saved in the 'models' repository. If you have already downloaded the 'models' directory during the installation step, you can skip to the following part to load the models.

## Loading the models
The following code can be used to load the models:

```
from modules.modeling.predict import load_model

MODELS_PATH = {
    'LogReg_Flatten': f'{MODELS_DIR}/LogReg_Flatten.pkl',
    'KNN_Flatten': f'{MODELS_DIR}/k-NN_Flatten.pkl',
    'SVM_Flatten': f'{MODELS_DIR}/SVM_Flatten.pkl',
    'LogReg_HOG': f'{MODELS_DIR}/LogReg_HOG.pkl',
    'KNN_HOG': f'{MODELS_DIR}/k-NN_HOG.pkl',
    'SVM_HOG': f'{MODELS_DIR}/SVM_HOG.pkl',
    'LogReg_SIFT': f'{MODELS_DIR}/LogReg_SIFT.pkl',
    'KNN_SIFT': f'{MODELS_DIR}/k-NN_SIFT.pkl',
    'SVM_SIFT': f'{MODELS_DIR}/SVM_SIFT.pkl'
}

models = {name: load_model(path) for name, path in MODELS_PATH.items()}
```

## Evaluating the models

The following code can be used to evaluate the models on the test set:

```
from modules.dataset import load_cifar10_dataset, get_random_subset
from modules.features import extract_features, standardize_features
from modules.config import DATA_DIR, MODELS_DIR

# Load the CIFAR-10 dataset
_, _, test_data, test_labels, label_names = load_cifar10_dataset(DATA_DIR)


test_data, test_labels = get_random_subset(test_data, test_labels, N_TEST_SAMPLES)

# Extract features from the test data
test_features_flatten = extract_features(test_data, method='flatten')
test_features_hog = extract_features(test_data, method='hog')
test_features_sift = extract_features(test_data, method='sift')

# Standardize the extracted features
test_features_flatten = standardize_features(test_features_flatten)
test_features_hog = standardize_features(test_features_hog)
test_features_sift = standardize_features(test_features_sift)


# Feature extraction methods
methods = {
    'Flatten': test_features_flatten,
    'HOG': test_features_hog,
    'SIFT': test_features_sift
}

# Initialize results dictionary
results = {}

# Load models and evaluate them on the test data
for method, test_features in methods.items():
    for name, path in MODELS_PATH.items():
        if method in name:
            print(f"Evaluating {name} model with {method} features...")
            model = load_model(path)
            predictions, accuracy, cm, report = test_model(model, test_features, test_labels, label_names)
            results[name] = {
                'predictions': predictions,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report
            }
```

Once you have stored the models results in the 'results' dictionary, you can use the following code to visualize the results:

```
from modules.plots import plot_accuracy, plot_roc_curve, plot_confusion_matrix

# Plot ROC curves
plot_roc_curve(results, test_labels)

# Plot accuracy comparison among models
plot_accuracy(results)

plot_confusion_matrix(results, 'SVM_HOG', label_names)
plot_confusion_matrix(results, 'LogReg_HOG', label_names)
```

## Prediction on a single image

The following code can be used to predict the label of a single image:

```
from modules.dataset import load_cifar10_dataset
from modules.config import DATA_DIR
from modules.modeling.predict import predict

# Load the CIFAR-10 dataset
train_data, train_labels, test_data, test_labels, label_names = load_cifar10_dataset(DATA_DIR)

image = test_data[0]
true_label = test_labels[0]
predicted_label = predict(image)

print(f"True Label: {label_names[true_label]}")
print(f"Predicted Label: {label_names[predicted_label]}")
```

The 'predict' function uses the 'SVM_HOG' model since it is our most accurate model. Therefore, you should make sure that the files 'SVM_HOG.pkl' and 'HOG_scaler.pkl' are present in the 'models' repository.