""" Implementation of a moving Gaussian function with an example. """

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt



def movingGaussian2D(data_tuple, a0, level_sum, sigma, x0, y0, L, omega, saturation_level=None):
    """ Moving Gaussian function with saturation intensity limiting.

    Based on:
        Peter Veres, Robert Jedicke, Larry Denneau, Richard Wainscoat, Matthew J. Holman and Hsing-Wen Lin
        Publications of the Astronomical Society of the Pacific
        Vol. 124, No. 921 (November 2012), pp. 1197-1207 

        The original equation given in the paper has a typo in the exp term, after sin(omega) there shoudl be 
        a minus, not a plus.


    Arguments:
        data_tuple: [tuple]
            - x: [ndarray] Array of X image coordinates.
            - y: [ndarray] Array of Y image coordiantes.
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
        
    x, y = data_tuple

    # Rotate the coordinates
    x_m = (x - x0)*np.cos(omega) - (y - y0)*np.sin(omega)
    y_m = (x - x0)*np.sin(omega) + (y - y0)*np.cos(omega)


    u1 = (x_m + L/2.0)/(sigma*np.sqrt(2))
    u2 = (x_m - L/2.0)/(sigma*np.sqrt(2))

    f1 = scipy.special.erf(u1) - scipy.special.erf(u2)

    # Evaluate the intensity at every pixel
    intens = a0 + level_sum/(2*sigma*np.sqrt(2*np.pi)*L)*np.exp(-y_m**2/(2*sigma**2))*f1


    # Limit intensity values to the given saturation limit
    if saturation_level is not None:
        intens[intens > saturation_level] = saturation_level


    return intens.ravel()




if __name__ == "__main__":

    # Image size
    x_size = 100
    y_size = 100

    # Generate the image range
    # x = np.linspace(-10, 10, x_size)
    # y = np.linspace(-10, 10, y_size)

    # xx, yy = np.meshgrid(x, y)

    yy, xx = np.indices((y_size, x_size))

    # Moving Gaussian parameters
    a0 = 10.0
    a1 = 100000.0
    sigma = 4.0
    x0 = 50.0
    y0 = 50.0
    L = 12.0
    omega = np.radians(45)
    saturation_level = 255


    # Evaluate the moving Gaussian
    img = movingGaussian2D((xx, yy), a0, a1, sigma, x0, y0, L, omega, saturation_level)
    img = img.reshape(x_size, y_size)


    # Add noise to the image where it is not saturating
    saturation_mask = img < saturation_level
    img[saturation_mask] += np.random.normal(0, 10.0, size=(x_size, y_size))[saturation_mask]
    img = np.clip(img, 0, saturation_level)


    ### Fit the moving Gaussian to fake data ###

    def movingGaussianResiduals_(params, x, y, saturation_level):
        return movingGaussian2D(x, *params, saturation_level=saturation_level) - y
        

    
    # Create x and y indices
    y_ind, x_ind = np.indices(img.shape)
    y_len, x_len = img.shape


    # Initial guess
    bg_est = np.percentile(img, 50.0)
    p0 = [bg_est, np.sum(img) - bg_est*y_len*x_len, 1.0, x_len/2.0, y_len/2.0, 1.0, 1.0]


    print('Initial guess:', p0)


    # Fit the moving Gaussian
    res = scipy.optimize.least_squares(movingGaussianResiduals_, p0, args=((y_ind, x_ind), img.ravel(), \
        saturation_level), loss='soft_l1')

    print(res)

    print('Fitted parameters:')
    print(res.x)


    ###########################################

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    # Plot the simulated image
    ax1.set_title('Image')
    ax1.imshow(img, vmin=0, vmax=saturation_level, cmap='gray')

    # Plot the region of saturation
    saturation_overlay = np.zeros_like(img) + saturation_level
    saturation_overlay = np.ma.masked_where(saturation_mask, saturation_overlay)
    ax1.imshow(saturation_overlay, cmap='hsv')

    # Plot the fit
    ax2.set_title('Fit')
    ax2.imshow(movingGaussian2D((y_ind, x_ind), *res.x, saturation_level=saturation_level).reshape(x_size, \
        y_size), vmin=0, vmax=saturation_level, cmap='gray')

    # Plot residuals
    ax3.set_title('Residuals')
    ax3.imshow(img - movingGaussian2D((y_ind, x_ind), *res.x, \
        saturation_level=saturation_level).reshape(x_size, y_size), vmin=0, vmax=saturation_level, cmap='gray')


    plt.tight_layout()
    
    plt.show()