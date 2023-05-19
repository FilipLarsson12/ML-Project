import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

import random
# TASK 1

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


# Plot function for each sample in the list samples, samples is a list containing weight samples
# generated from a weight distribution:
def plot_function_for_each_sample(samples, points_added):
    x = np.linspace(-2, 2, 100)
    i = 1
    for sample in samples:
        w0 = sample[0]
        w1 = sample[1]
        t = w0*x + w1
        plt.plot(x, t, label=f'Line {i}')
        i += 1

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'Plot of f(x) for different weights, based on {points_added} point/points.')
    plt.grid(True)
    plt.show()

def add_points(point_amount, w_distribution):
    w_prior = w_distribution
    w_posterior = 0

    points_added = 0
    for i in range(point_amount):
        if (i == 0): 
            points_added += 1
            random_x = random.uniform(-1, 1)
            t = phi_of_x(random_x)
            w_posterior = calculate_posterior(random_x, w_prior)
            samples = w_posterior.rvs(size=5)
            plot_w(w_posterior)
            plot_function_for_each_sample(samples, points_added)
        else:
            points_added += 1
            random_x = random.uniform(-1, 1)
            t = phi_of_x(random_x)
            w_posterior = calculate_posterior(random_x, w_posterior)
            samples = w_posterior.rvs(size=5)
            plot_w(w_posterior)
            plot_function_for_each_sample(samples, points_added)




# TASK 2

# Uppgift 1
# x är en vektor med 2 element där varje element ingår i: (-1,1).

def generate_input_space():
    X1 = np.arange(-1, 1.05, 0.05)
    X2 = np.arange(-1, 1.05, 0.05)
    return X1, X2

def calculate_t(x1, x2):
    epsilon = multivariate_normal(0, 0.2)
    w = [0, 1.5, -0,8]
    t = w[0] + w[1]*x1 + w[2]*x2 + epsilon.rvs()
    return t


#def calculate_log_likelihood():



def main():
    # Lista som håller koll på alla punkter som vi studerat och varje gång en 
    # punkt läggs till räknar vi ut w_posterior.
    w_prior = calculate_w_prior()
    plot_w(w_prior)
    '''
    points_added = []
    w_posterior1 = calculate_posterior(0.9, calculate_w_prior())

    points_added.append([0.9, phi_of_x(0.9)])

    # Uppgift 3
    samples = w_posterior1.rvs(size=5)
    # plot_function_for_each_sample(samples, len(points_added))

    # Uppgift 4: Adding additional data points:
    add_points(19, w_posterior1)
    #plot_function_for_each_sample(samples, len(points_added))

    print(points_added)
    '''
    

    
 
    # Uppgift 5 
    '''
    Each time we add a data point for the function f(x) we get an evermore accurate estimation of the
    weights w0 and w1 which values are 0.5 and -1.5. This is represented in the visual respresentation of
    the weights, we can see that the distribution generally gets smaller and smaller and it is centered 
    around the point: (0.5, -1.5). We can also see that after we have calculated the posterior distribution
    of the weights for each new datapoint we generated we get functions f(x) which more and more resemble the
    true function with the correct weights which is: f(x) = 0.5*x -1.5 + epsilon. This makes sense because 
    we get better estimations of the weight distribution which in turn gives better estimations for the 
    function f(x). So our interpretation of this effect is that the more datapoints we give to our program
    the better the program gets at estimating the true weights and the true function f(x).
    '''

    # Uppgift 6
    '''
    Starting to test with sigma_squared = 0.1
    Observing that when sigma_squared is 0.1 which is smaller than 0.2 we get more accurate estimations of 
    the weights and functions quicker. This makes sense because there is less random variation in the 
    calculation of w_posterior which in turn give a better and more accurate estimation. The effect 
    on the posterior is therefore positive.
    Testing with sigma squared = 0.4:
    The effect on the posterior is the reverse of when sigma_squared was 0.1, it is not as precise as
    before, the estimation of the posterior is not as confident. The effect on the posterior is negative.
    Testing with sigma squared = 0.8:
    Same effect as when sigma_squared was 0.4 bet even more pronounced. The distribution of the posterior
    is even more spread out than before. Our model account okay for various levels of noise but there is a
    clear and visible difference when the sigma_squared variable have different values.
    '''


main()