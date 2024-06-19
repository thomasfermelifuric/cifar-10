import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from modules.dataset import load_cifar10_dataset, get_random_subset
from modules.features import extract_features, standardize_features
from modules.config import DATA_DIR, MODELS_DIR, N_TRAIN_SAMPLES
from modules.plots import show_distribution

# Ensure the MODELS_DIR exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the CIFAR-10 dataset
train_data, train_labels, _, _, label_names = load_cifar10_dataset(DATA_DIR)

train_data, train_labels = get_random_subset(train_data, train_labels, N_TRAIN_SAMPLES)
show_distribution(train_labels, label_names)
print("Dataset loaded")

# Classifier definitions
classifiers = {
    'LogReg': LogisticRegression(max_iter=10000),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', gamma='scale')
}


# Training function
def train_and_save(classifier, train_features, train_labels, model_name):
    classifier.fit(train_features, train_labels)
    # Save the trained model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)

# Extract features
print("Extracting Features...")
train_features_flatten = extract_features(train_data, method='flatten')
train_features_hog = extract_features(train_data, method='hog')
train_features_sift = extract_features(train_data, method='sift')
print("Features Extracted")

# Feature extraction methods
methods = {
    'Flatten': train_features_flatten,
    'HOG': train_features_hog,
    'SIFT': train_features_sift
}

# Iterate over feature extraction methods
for method, train_features in methods.items():
    # Standardize features
    print(f"Standardizing Features for {method}...")
    train_features = standardize_features(train_features)

    # Iterate over classifiers
    for name, classifier in classifiers.items():
        print(f"Training {name} classifier with {method} features...")
        model_name = f"{name}_{method}"
        train_and_save(classifier, train_features, train_labels, model_name)
