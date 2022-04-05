'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import scipy.special as sp
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        curr_class = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(curr_class, axis=0)
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(0, 10):
        curr_class = data.get_digits_by_label(train_data, train_labels, i)
        n = np.transpose((curr_class - means[i])).dot((curr_class - means[i])) / curr_class.shape[0]
        covariances[i] = n + (0.01 * np.eye(64))
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    det_var = np.linalg.det(covariances)
    log_pi = np.log(2 * np.pi)
    log_det = np.log(det_var)
    inv_var = np.linalg.inv(covariances)
    p = np.zeros((digits.shape[0], 10))
    for i in range(digits.shape[0]):
        x = digits[i]
        for k in range(10):
            p[i][k] = -32 * log_pi - 1 / 2 * log_det[k] - 1 / 2 * ((x - means[k]).T.dot(inv_var[k])).dot(x - means[k])
    return p


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    gen_likelihood = generative_likelihood(digits, means, covariances)
    plus_part = gen_likelihood + np.log(1 / 10)
    minus_part = sp.logsumexp(gen_likelihood + np.log(1 / 10))
    return plus_part - minus_part


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    total = 0
    for i in range(digits.shape[0]):
        total += cond_likelihood[i][int(labels[i])]
    # Compute as described above and return
    return total / digits.shape[0]


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    most_likely = np.argmax(cond_likelihood, axis=1)
    return most_likely


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    train_like = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_like = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    train_predict = classify_data(train_data, means, covariances)
    test_predict = classify_data(test_data, means, covariances)

    train_correct = 0
    test_correct = 0
    for k in range(0, train_data.shape[0]):
        if train_predict[k] == train_labels[k]:
            train_correct += 1
    train_acc = train_correct / train_data.shape[0]
    for k in range(0, test_data.shape[0]):
        if test_predict[k] == test_labels[k]:
            test_correct += 1
    test_acc = test_correct / test_data.shape[0]

    print("Average conditional probability for training set:" + str(train_like))
    print("Average conditional probability for testing set:" + str(test_like))
    print("Accuracy for training set is: " + str(train_acc))
    print("Accuracy for testing set is: " + str(test_acc))

    for k in range(0, 10):
        eigen_values, eigen_vectors = np.linalg.eig(covariances[k])
        max_eigen_vector_index = np.argmax(eigen_values)
        max_eigen_vector = eigen_vectors[:, max_eigen_vector_index].reshape(8, 8)
        plt.figure(1)
        plt.subplot(2, 5, k + 1)
        plt.imshow(max_eigen_vector, cmap="gray")
        plt.xlabel("Label = {}".format(k))
    plt.savefig("All_plots")


if __name__ == '__main__':
    main()
