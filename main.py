import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


x = np.arange(-1, 1.01, 0.01).reshape(1,-1)

epsilon_mean = 0
epsilon_standard_deviation = np.sqrt(0.2)
epsilon = multivariate_normal([0], [[0.2]])

# Uppgift 1
# Denna funktion skapar en plot som illustrerar vars sannolikhetsmassan ligger för värdena på w0 och w1:


def calculate_w_prior():
    w_dimension = 2
    alpha = 0.2
    mean = np.zeros(w_dimension)
    covariance = np.eye(w_dimension) / alpha
    prior_w = multivariate_normal(mean, covariance)
    return prior_w

def plot_w(w):

    print(w.rvs())
    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    pos = np.dstack((x, y))

    # Compute the density function for each point in the grid
    density = w.logpdf(pos)

    # Make a contour plot
    plt.contourf(x, y, density, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Contour plot of Multivariate Normal Distribution')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()


# Uppgift 2:

def phi_of_x(x):
    t = 0.5*x -1.5
    return t


def calculate_likelihood(single_x):
    prior_w = calculate_w_prior()
    w_value = prior_w.rvs()
    w_value.transpose()
    likelihood_mean = w_value*phi_of_x(single_x)
    likelihood_variance = 1 / np.sqrt(0.2)
    likelihood = multivariate_normal(likelihood_mean, likelihood_variance)
    return likelihood

def calculate_posterior(single_x):
    


    w_posterior = likelihood.pdf
    return w_posterior





def main():
    w_posterior = calculate_posterior(0.5)
    #plot_w(w_posterior)
    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    pos = np.dstack((x, y))

    # Compute the density function for each point in the grid
    density = w_posterior.logpdf(pos)

    # Make a contour plot
    plt.contourf(x, y, density, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Contour plot of Multivariate Normal Distribution')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()



main()