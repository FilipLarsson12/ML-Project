import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# Uppgift 2.4



def calculate_w_prior(alpha):
    w_dimension = 3
    mean = np.zeros(w_dimension)
    covariance = np.eye(w_dimension) / alpha
    prior_w = multivariate_normal(mean, covariance)
    return prior_w

def calculate_t(x1, x2, variance):
    epsilon = multivariate_normal(0, variance)
    w = [0, 1.5, -0.8]
    t = w[0] + w[1]*x1 + w[2]*x2 + epsilon.rvs()
    return t

def generate_all_data(variance):
    T_values = []
    X1 = np.arange(-0.3, 1.05, 0.05)
    X2 = np.arange(-0.3, 1.05, 0.05)
    input_matrix = np.array([[x1, x2] for x1 in X1 for x2 in X2])
    for X in input_matrix:
        t = calculate_t(X[0], X[1], variance)
        T_values.append(t)
    return input_matrix, T_values


def calculate_posterior(x1, x2, w_prior, variance):
    X = np.array([1, x1, x2])
    t = calculate_t(x1, x2, variance)

    Sigma_prior = w_prior.cov
    mu_prior = w_prior.mean

    XTX = np.outer(X, X)
    XTy = X * t

    Sigma_posterior = np.linalg.inv(np.linalg.inv(Sigma_prior) + XTX / variance)
    mu_posterior = Sigma_posterior @ (np.linalg.inv(Sigma_prior) @ mu_prior + XTy / variance)
    posterior = multivariate_normal(mean=mu_posterior, cov=Sigma_posterior)
    return posterior

def generate_final_distribution(variance, w_prior, input_data):
    current_distribution = w_prior
    for X in input_data:
        current_distribution = calculate_posterior(X[0], X[1], current_distribution, variance)
    return current_distribution
    




def plot_posterior(w_posterior, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    w0, w1, w2 = np.random.multivariate_normal(w_posterior.mean, w_posterior.cov, 5000).T
    ax.scatter(w0, w1, w2, alpha=0.6, edgecolors='w', s=20)
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
    ax.set_zlabel('w2')
    plt.title(title)
    plt.show()


def main():
    variance_list = [0.2, 0.4, 0.6]
    alpha_list = [0.7, 1.5, 3.0]
    all_inputdata_for_w1, all_outputdata_for_w1 = generate_all_data(variance_list[0])
    all_inputdata_for_w2, all_outputdata_for_w2 = generate_all_data(variance_list[1])
    all_inputdata_for_w3, all_outputdata_for_w3 = generate_all_data(variance_list[2])
    w_prior1 = calculate_w_prior(alpha_list[0])
    w_prior2 = calculate_w_prior(alpha_list[1])
    w_prior3 = calculate_w_prior(alpha_list[2])
    w_posterior1 = generate_final_distribution(variance_list[0], w_prior1, all_inputdata_for_w1)
    w_posterior2 = generate_final_distribution(variance_list[1], w_prior2, all_inputdata_for_w2)
    w_posterior3 = generate_final_distribution(variance_list[2], w_prior3, all_inputdata_for_w3)


    plot_posterior(w_posterior1, f'w_posterior1. variance is: {variance_list[0]}.')
    plot_posterior(w_posterior2, f'w_posterior2. variance is: {variance_list[1]}.')
    plot_posterior(w_posterior3, f'w_posterior3. variance is: {variance_list[2]}.')

main()