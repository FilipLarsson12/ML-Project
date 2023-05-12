import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

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
    print(prior_w)
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

def calculate_likelihood(w, x, t):
    phiOfX = [x, 1]
    mu = w[0]*phiOfX[0] + w[1]*phiOfX[1] # Väntevärdet för likelihood-funktionen
    sigma = 0.2 # Variansen för likelihood-funktionen
    likelihood = multivariate_normal(mu, sigma).pdf(t)
    return likelihood


# Formel 10 och lite 11 viktig för att räkna ut likelihood för flera punkter.

# Definition för Phi finns på formel 14

def calculate_posterior(single_x):
    w_prior = calculate_w_prior()
    mu_prior = w_prior.mean()
    sigma_prior = w_prior.cov()
    sigma_squared = 0.2
    real_t = phi_of_x(single_x)
    #likelihood = calculate_likelihood(w_prior.rvs(2), single_x, real_t)
    x_vector = [single_x, 1]
    XTX = np.dot(x_vector.T, x_vector)
    XTy = np.dot(x_vector.T, real_t)
    sigma_posterior = np.linalg.inv(np.linalg.inv(sigma_prior) + XTX / sigma_squared)
    mu_posterior = np.dot(sigma_posterior, (np.dot(np.linalg.inv(sigma_prior), mu_prior) + XTy / sigma_squared))
    posterior_w = multivariate_normal(mu_posterior, sigma_posterior)
    return posterior_w
    
    







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