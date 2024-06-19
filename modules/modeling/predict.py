import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from modules.config import MODELS_DIR
from modules.features import extract_features, standardize_features


# Load saved models
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Function to load the saved scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler


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


# Function to test a model
def test_model(model, test_features, test_labels, label_names):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=label_names)
    return predictions, accuracy, cm, report


# Function to predict the class of an image using the SVM_HOG model
def predict(image):
    # Load the SVM_HOG model
    model_path = MODELS_PATH['SVM_HOG']
    scaler_path = f'{MODELS_DIR}/HOG_scaler.pkl'

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Extract HOG features from the image
    hog_features = extract_features([image], method='hog')

    # Standardize the features
    hog_features = scaler.transform(hog_features)

    # Predict the class
    prediction = model.predict(hog_features)

    return prediction[0]


# Save the results dictionary to a file
def save_results(results, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)


def load_results(file_path):
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
    return results
