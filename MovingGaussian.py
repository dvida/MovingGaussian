""" Implementation of a moving Gaussian function with an example. """

from __future__ import division

import numpy as np
import scipy.special
import matplotlib.pyplot as plt



def movingGaussian2D(x, y, a0, level_sum, sigma, x0, y0, L, omega, saturation_level=None):
    """ Moving Gaussian function with saturation intensity limiting.

    Based on:
        Peter Veres, Robert Jedicke, Larry Denneau, Richard Wainscoat, Matthew J. Holman and Hsing-Wen Lin
        Publications of the Astronomical Society of the Pacific
        Vol. 124, No. 921 (November 2012), pp. 1197-1207 

        The original equation given in the paper has a typo in the exp term, after sin(omega) there shoudl be 
        a minus, not a plus.


    Arguments:
        x: [ndarray] Array of X image coordinates.
        y: [ndarray] Array of Y image coordiantes.
        a0: [float] Background level.
        level_sum: [float] Total flux of the Gaussian.
        sigma: [float] Standard deviation.
        x0: [float] X coordinate of the centre of the track.
        y0: [float] Y coordinate of the centre of the track.
        L: [float] Length of the track.
        omega: [float] Angle of the track.

    Keyword arguments:
        saturation_level: [float] Level of saturation. None by default.

    """
    
    # Rotate the coordinates
    x_m = (x - x0)*np.cos(omega) - (y - y0)*np.sin(omega)
    y_m = (x - x0)*np.sin(omega) + (y - y0)*np.cos(omega)


    u1 = (x_m + L/2.0)/(sigma*np.sqrt(2))
    u2 = (x_m - L/2.0)/(sigma*np.sqrt(2))

    f1 = scipy.special.erf(u1) - scipy.special.erf(u2)

    # Ealuate the intensity at every pixel
    intens = a0 + level_sum/(2*sigma*np.sqrt(2*np.pi)*L)*np.exp(-y_m**2/(2*sigma**2))*f1


    # Limit intensity values to the given saturation limit
    if saturation_level is not None:
        intens[intens > saturation_level] = saturation_level


    return intens




if __name__ == "__main__":

    # Image size
    x_size = 100
    y_size = 100

    # Generate the image range
    x = np.linspace(-10, 10, x_size)
    y = np.linspace(-10, 10, y_size)

    xx, yy = np.meshgrid(x, y)

    # Moving Gaussian parameters
    a0 = 10.0
    a1 = 7000.0
    sigma = 1.0
    x0 = 0.0
    y0 = 0.0
    L = 7.0
    omega = np.radians(45)
    saturation_level = 255


    # Evaluate the moving Gaussian
    img = movingGaussian2D(xx, yy, a0, a1, sigma, x0, y0, L, omega, saturation_level)
    img.reshape(x_size, y_size)


    plt.imshow(img)
    plt.show()