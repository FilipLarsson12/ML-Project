import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

# Uppgift 1
def calculate_w_prior():
    w_dimension = 2
    alpha = 0.2
    mean = np.zeros(w_dimension)
    covariance = np.eye(w_dimension) / alpha
    prior_w = multivariate_normal(mean, covariance)
    return prior_w

def plot_w(w):
    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    pos = np.dstack((x, y))
    density = w.pdf(pos)
    plt.contourf(x, y, density, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Contour plot of Multivariate Normal Distribution')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()

# Uppgift 2:

def phi_of_x(x):
    t = 0.5 * x - 1.5
    return t

def calculate_likelihood(mu, t):
    sigma = 0.2  # Variansen för likelihood-funktionen
    likelihood = norm(mu, np.sqrt(sigma)).pdf(t)
    return likelihood

def calculate_posterior(single_x, w_prior):
    X = np.array([single_x, 1])
    t = phi_of_x(single_x)
    sigma_squared = 0.2

    Sigma_prior = w_prior.cov
    mu_prior = w_prior.mean

    XTX = np.outer(X, X)
    XTy = X * t

    Sigma_posterior = np.linalg.inv(np.linalg.inv(Sigma_prior) + XTX / sigma_squared)
    mu_posterior = Sigma_posterior @ (np.linalg.inv(Sigma_prior) @ mu_prior + XTy / sigma_squared)
    
    likelihood = calculate_likelihood(np.dot(mu_posterior, X), t)

    posterior = multivariate_normal(mean=mu_posterior, cov=Sigma_posterior)
    return posterior

def plot_function_for_each_sample(samples):
    x = np.linspace(-2, 2, 100)
    i = 1
    for sample in samples:
        w0 = sample[0]
        w1 = sample[1]
        t = w0*x + w1
        plt.plot(x, t, label=f'Line {i+1}')
        i += 1

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Plot of f(x) for different weights')
    plt.grid(True)
    plt.show()



def main():

    w_posterior1 = calculate_posterior(0.9, calculate_w_prior())
    
    samples = w_posterior1.rvs(size=10)
    for sample in samples:
        print(sample)
    plot_function_for_each_sample(samples)
    plot_w(w_posterior1)
    


main()