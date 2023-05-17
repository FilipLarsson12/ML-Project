from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def generate_input_space():
    X1 = np.arange(-1, 1.05, 0.05)
    X2 = np.arange(-1, 1.05, 0.05)
    input_matrix = np.array([[x1, x2] for x1 in X1 for x2 in X2])
    return input_matrix

def calculate_t(x1, x2, variance):
    epsilon = multivariate_normal(0, variance)
    w = [0, 1.5, -0.8]
    t = w[0] + w[1]*x1 + w[2]*x2 + epsilon.rvs()
    return t

def generate_seen_data(variance):
    T_values = []
    X1 = np.arange(-0.3, 0.35, 0.05)
    X2 = np.arange(-0.3, 0.35, 0.05)
    input_matrix = np.array([[x1, x2] for x1 in X1 for x2 in X2])
    for X in input_matrix:
        t = calculate_t(X[0], X[1], variance)
        T_values.append(t)
    T_values.reverse()
    return input_matrix, T_values

def generate_unseen_data(variance):
    T_values = []
    X1_first_half = np.arange(-1.00, -0.30, 0.05)
    X1_second_half = np.arange(0.35, 1.05, 0.05)
    X2_first_half = np.arange(-1.00, -0.30, 0.05)
    X2_second_half = np.arange(0.35, 1.05, 0.05)
    X1 = np.concatenate(X1_first_half, X1_second_half)
    X2 = np.concatenate(X2_first_half, X2_second_half)

    input_matrix = np.array([[x1, x2] for x1 in X1 for x2 in X2])
    for X in input_matrix:
        t = calculate_t(X[0], X[1], variance)
        T_values.append(t)
    T_values.reverse()
    return input_matrix, T_values


def calculate_maximum_likelihood_weights(input_matrix_training_data, T_values_training_data):
    # No need to reshape input_matrix_training_data, it should already be of shape (n, 2)

    ones = np.ones((input_matrix_training_data.shape[0], 1))
    phi_matrix = np.concatenate((ones, input_matrix_training_data), axis=1)
    phi_matrix_transposed = phi_matrix.T

    weights = np.linalg.inv(phi_matrix_transposed @ phi_matrix) @ phi_matrix_transposed @ T_values_training_data
    return weights



def main():
    variance = 0.2
    real_weights = [0, 1.5, -0.8]
    # Uppgift 1:
    '''
    input_matrix = generate_input_space()
    T_values = []
    for X in input_matrix:
        t = calculate_t(X[0], X[1], variance)
        T_values.append(t)
    T_values.reverse()
    '''


    input_matrix, T_values = generate_seen_data(variance)
    weights = calculate_maximum_likelihood_weights(input_matrix, T_values)
    print(real_weights, weights)

    '''
    input_matrix, T_values = generate_seen_data(variance)

    X1 = input_matrix[:, 0]
    X2 = input_matrix[:, 1]
    T_values = np.array(T_values)
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X1, X2, T_values)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('t')

    ax.set_title('3D scatter plot with variance = {}'.format(variance))


    plt.show()
    '''

main()