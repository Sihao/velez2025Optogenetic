import numpy as np
import scipy.stats as stats
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def classify_stims(traces_stim_avg, labels, test_size=0.2, random_state=42):
    """Classify stimuli based on average traces using linear SVM with L1 regularization.
    Perform k-fold cross validation on the training set and average the weights. Per fold
    the best regularization parameter C is chosen based on the validation set (swept from 0.1 to 1).

    Parameters
    ----------
    traces_stim_avg : np.ndarray
        Average response for each neuron. Rows are neurons, column contains response amplitude. Shape is
        (n_neurons, n_trials X n_stimuli).
    labels : np.ndarray
        Stimulus labels for each column in traces_stim_avg.
    test_size : float
        Proportion of data to hold out for testing.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    clf : sklearn.svm.LinearSVC
        Trained classifier.
    accuracy : float
        Accuracy of the classifier on the test set.
    X_test : np.ndarray
        Test set.
    y_test : np.ndarray
        Test labels.
    y_pred : np.ndarray
        Predicted labels.
    """

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(traces_stim_avg.T, labels, test_size=test_size, random_state=random_state)

    # Clean NaNs
    nan_idx = np.isnan(X_train).any(axis=1)
    X_train = X_train[~nan_idx]
    y_train = y_train[~nan_idx]

    nan_idx = np.isnan(X_test).any(axis=1)
    X_test = X_test[~nan_idx]
    y_test = y_test[~nan_idx]

    # Split train into train-validation and k-fold cross validation
    cross_val_weights = []
    n_class_members = np.min(np.unique(y_train, return_counts=True)[1])
    n_splits = np.min([n_class_members, 16]) # Limit number of splits to 16, saves time for high trial experiments
    skf = StratifiedShuffleSplit(n_splits=n_splits, random_state=42, test_size=0.2)

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_train, X_train_val = X_train[train_index], X_train[val_index]
        y_train_train, y_train_val = y_train[train_index], y_train[val_index]

        # Train SVM
        # Sweep sparsity parameter C from 0.1 to 1
        c = np.linspace(0.1, 1, 10)
        c_sweep_clfs = []
        for c_ in c:
            clf = svm.LinearSVC(dual='auto', penalty='l1', C=c_,
                                random_state=42, max_iter=100000, tol=1e-6)
            clf.fit(X_train_train, y_train_train)
            c_sweep_clfs.append(clf)

        # Choose best C
        accuracies = [accuracy_score(y_train_val, clf.predict(X_train_val)) for clf in c_sweep_clfs]
        cross_val_weights.append(c_sweep_clfs[np.argmax(accuracies)].coef_)

    # Average weights
    cross_val_weights = np.mean(np.array(cross_val_weights), axis=0)

    # Assign weights to clf
    clf.coef_ = cross_val_weights

    # Predict
    y_pred = clf.predict(X_test)

    return clf, X_test, y_test, y_pred


def classify_stims_mlp(traces_stim_avg, labels, test_size=0.2, random_state=42):
    """Classify stimuli based on average traces using a multi-layer perceptron with 1 hidden layer.
    Perform k-fold cross validation on the training set and average the weights. Per fold
    the best regularization parameter C is chosen based on the validation set (swept from 0.1 to 1).

    Parameters
    ----------
    traces_stim_avg : np.ndarray
        Average response for each neuron. Rows are neurons, column contains response amplitude. Shape is
        (n_neurons, n_trials X n_stimuli).
    labels : np.ndarray
        Stimulus labels for each column in traces_stim_avg.
    test_size : float
        Proportion of data to hold out for testing.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    clf : sklearn.neural_network.MLPClassifier
        Trained classifier.
    accuracy : float
        Accuracy of the classifier on the test set.
    X_test : np.ndarray
        Test set.
    y_test : np.ndarray
        Test labels.
    y_pred : np.ndarray
        Predicted labels.
    """

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(traces_stim_avg.T, labels, test_size=test_size, random_state=random_state)

    # Clean NaNs
    nan_idx = np.isnan(X_train).any(axis=1)
    X_train = X_train[~nan_idx]
    y_train = y_train[~nan_idx]

    nan_idx = np.isnan(X_test).any(axis=1)
    X_test = X_test[~nan_idx]
    y_test = y_test[~nan_idx]

    # Split train into train-validation and k-fold cross validation
    clfs = []

    n_class_members = np.min(np.unique(y_train, return_counts=True)[1])
    n_splits = np.min([n_class_members, 16]) # Limit number of splits to 16, saves time for high trial experiments
    skf = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)

    n_classes = len(np.unique(y_train))
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_train, X_train_val = X_train[train_index], X_train[val_index]
        y_train_train, y_train_val = y_train[train_index], y_train[val_index]

        # Train MLP
        clf = MLPClassifier(hidden_layer_sizes=(n_classes*4, n_classes*2, n_classes),
                            max_iter=100000, random_state=42,
                            activation='relu', learning_rate_init=1e-3)
        clf.fit(X_train_train, y_train_train)

        clfs.append(clf)

    # Average weights (use last)
    for i, _ in enumerate(clf.coefs_):
        cross_val_weights = np.mean([clf.coefs_[i] for clf in clfs], axis=0)
        clf.coefs_[i] = cross_val_weights

    # Predict
    y_pred = clf.predict(X_test)

    # Validate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {accuracy}')
    # Scale accuracy with number of classes
    n_unique_labels = len(np.unique(labels))
    scaled_accuracy = (accuracy - 1 / n_unique_labels) / (1 - 1 / n_unique_labels)
    print(f'Scaled accuracy: {scaled_accuracy}')

    return clf, X_test, y_test, y_pred


def compute_neighbour_correlation(traces_trial_avg, nm_identities):
    """Compute correlation between responses of neighbouring neuromasts. If neuromast has two neighbours, the
    correlation is computed as the average of the correlation between the response of the neuromast and the
    response of each of its neighbours. If neuromast has only one neighbour, the correlation is computed
    between the response of the neuromast and the response of its neighbour.

    Parameters
    ----------
    traces_trial_avg : np.ndarray
        Average response for each neuron. Rows are neurons, column contains response amplitude. Shape is
        (n_neurons, n_samples, n_stimuli).
    nm_identities : np.ndarray
        Neuromast identities for each neuron. Shape is (n_stim - 1). Assuming the last stimulus is the
        visual control stimulus. Sometimes there are two Ter neuromasts, in which case shape is (n_stim - 2).

    Returns
    -------
    neighbour_correlation : dict
        Dictionary containing the correlation between responses of neighbouring neuromasts. Keys are the
        neuromast identities and values are the correlation coefficient with the neighbours.
    """
    # Get number of neurons
    n_neurons = traces_trial_avg.shape[0]

    # Get number of stimuli
    n_stim = traces_trial_avg.shape[-1]

    # Get number of neuromasts
    n_nm = n_stim - 1

    # Create dictionary to store correlation coefficients
    neighbour_correlation = {}

    # Loop over neuromasts
    for nm in range(n_nm):
        nm_id = nm_identities[nm]
        # Loop over neurons
        for neuron in range(n_neurons):
            # Get response of neuromast
            nm_response = traces_trial_avg[neuron, :, nm]

            if nm == 0:
                # Get response of neighbour
                neighbour_response = traces_trial_avg[neuron, :, nm + 1]

                # Compute correlation
                correlation = stats.pearsonr(nm_response, neighbour_response, alternative='greater')
                if correlation.pvalue > 0.05:
                    correlation = 0
                else:
                    correlation = correlation.statistic
                # Store correlation
                if nm_id in neighbour_correlation:
                    neighbour_correlation[nm_identities[nm]].append(correlation)
                else:
                    neighbour_correlation[nm_identities[nm]] = [correlation]
            elif nm == n_nm - 1:
                # Get response of neighbour
                neighbour_response = traces_trial_avg[neuron, :, nm - 1]

                # Compute correlation
                correlation = stats.pearsonr(nm_response, neighbour_response, alternative='greater')
                if correlation.pvalue > 0.05:
                    correlation = 0
                else:
                    correlation = correlation.statistic

                # Store correlation
                if nm_id in neighbour_correlation:
                    neighbour_correlation[nm_identities[nm]].append(correlation)
                else:
                    neighbour_correlation[nm_identities[nm]] = [correlation]
            else:
                # Get response of neighbour
                neighbour_response_1 = traces_trial_avg[neuron, :, nm - 1]
                neighbour_response_2 = traces_trial_avg[neuron, :, nm + 1]

                # Compute correlation
                correlation_1 = stats.pearsonr(nm_response, neighbour_response_1, alternative='greater')
                correlation_2 = stats.pearsonr(nm_response, neighbour_response_2, alternative='greater')

                if correlation_1.pvalue > 0.05:
                    correlation_1 = 0
                else:
                    correlation_1 = correlation_1.statistic
                if correlation_2.pvalue > 0.05:
                    correlation_2 = 0
                else:
                    correlation_2 = correlation_2.statistic

                correlation = (correlation_1 + correlation_2) / 2

                # Store correlation
                if nm_id in neighbour_correlation:
                    neighbour_correlation[nm_identities[nm]].append(correlation)
                else:
                    neighbour_correlation[nm_identities[nm]] = [correlation]

    return neighbour_correlation