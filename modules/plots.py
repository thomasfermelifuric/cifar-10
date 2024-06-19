from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def visualize_flattened_image(image, flattened_image):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Flattened Image")
    plt.plot(flattened_image)
    plt.axis('off')

    plt.show()


def visualize_hog_features(image, hog_image):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("HOG Features")
    plt.imshow(hog_image, cmap='gray')
    plt.axis('off')

    plt.show()


def visualize_sift_features(image, keypoints):
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("SIFT Features")
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


def plot_roc_curve(results, test_labels, n_classes=10):
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan'])

    for (model_name, result), color in zip(results.items(), colors):
        if 'SVM' in model_name or 'LogReg' in model_name:
            y_score = result['predictions']

            # Ensure y_score is 2D
            if y_score.ndim == 1:
                y_score = label_binarize(y_score, classes=range(n_classes))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(label_binarize(test_labels, classes=range(n_classes))[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(label_binarize(test_labels, classes=range(n_classes)).ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            plt.plot(fpr["micro"], tpr["micro"], color=color, lw=2, label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_accuracy(results):
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]

    plt.figure()
    plt.barh(model_names, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Accuracy Comparison Among Models')
    plt.show()


def plot_confusion_matrix(results, model_name, label_names):
    cm = results[model_name]['confusion_matrix']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


def show_distribution(labels, label_names):
    class_counts = np.bincount(labels, minlength=len(label_names))

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_names, y=class_counts, palette='viridis')
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Distribution of selected images')
    plt.xticks(rotation=45)
    plt.show()
