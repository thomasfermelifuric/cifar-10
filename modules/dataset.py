import pickle
import numpy as np


def unpickle(file):
    """Load a pickled file."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10_dataset(data_dir):
    """Load the CIFAR-10 dataset."""
    # Load the batch meta file to get label names
    meta = unpickle(f'{data_dir}/batches.meta')
    label_names = [label.decode('utf-8') for label in meta[b'label_names']]

    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(f'{data_dir}/data_batch_{i}')
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']

    # Convert to numpy arrays
    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    # Load test data
    test_batch = unpickle(f'{data_dir}/test_batch')
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    # Reshape and return the data
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return train_data, train_labels, test_data, test_labels, label_names


def get_random_subset(data, labels, n):
    indices = np.random.choice(len(labels), n, replace=False)
    subset_data = data[indices]
    subset_labels = labels[indices]
    return subset_data, subset_labels
