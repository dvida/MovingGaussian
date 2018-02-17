""" Fitting a moving Gaussian to fireball data. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


from FRbin import read as readFR
from MovingGaussian import movingGaussian2D


def centroidImage(img):
    """ Find the centroid on the given image. """

    # Estimate the background as a median
    bg_lvl = np.percentile(img.ravel(), 25)

    x_sum = 0
    y_sum = 0

    # Calculcate the centroids
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            y_sum += i*(img[i, j] - bg_lvl)
            x_sum += j*(img[i, j] - bg_lvl)


    weight_sum = np.sum(img) - bg_lvl*img.shape[0]*img.shape[1]

    x_cent = x_sum/weight_sum
    y_cent = y_sum/weight_sum

    return x_cent, y_cent



def fitMovingGaussian(img, x_cent, y_cent, saturation_level=255):
    """ Fit the moving gaussian function to the given image with the given centroids. """

    ### Fit the moving Gaussian to fake data ###

    def movingGaussianResiduals_(params, x, y, saturation_level):
        return movingGaussian2D(x, *params, saturation_level=saturation_level) - y
        

    
    # Create x and y indices
    y_ind, x_ind = np.indices(img.shape)
    y_len, x_len = img.shape


    # Initial guess
    bg_est = np.percentile(img, 25)
    p0 = [bg_est, np.sum(img) - bg_est*y_len*x_len, 1.0, y_cent, x_cent, 1.0, np.pi]


    print('Initial guess:', p0)


    # Init the bounds
    #             a0, level_sum, sigma,  x0,     y0,     L,     omega
    bounds = ([     0,      0,      0,      0,      0,      0,       0], \
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2*np.pi])

    # Fit the moving Gaussian
    res = scipy.optimize.least_squares(movingGaussianResiduals_, p0, args=((y_ind, x_ind), img.ravel(), \
        saturation_level), loss='huber', bounds=bounds)

    print(res)

    print('Fitted parameters:')
    print(res.x)


    ###########################################

    return res.x



if __name__ == "__main__":

    # Data directory
    dir_path = 'FR_data'

    # FR bin path
    file_path = "FR_HR0002_20180215_231005_927_0542208.bin"
    #file_path = "FR_HR0002_20171231_011845_123_0250368.bin"
    #file_path = "FR_HR0002_20171231_011855_362_0250624.bin"

    saturation_level = 255


    # Load the FR bin file
    fr_list = readFR(dir_path, file_path)


    for line in fr_list.frames:

        flux_fit_list = []
        flux_sum_list = []

        for i, frame in enumerate(line):

            print('Frame No:', i)

            # # TEST
            # if i != 13:
            #     continue

            # # Fit only frames which are saturating (more than 4 pixels)
            # if np.where(frame == saturation_level)[0].size < 4:
            #     continue

            # Find the centroid of the fireball
            x, y = centroidImage(frame)

            # Skip the fit if the centroid is not +/-25% off the centre
            if ((abs(x/frame.shape[1] - 0.5)) > 0.25) or ((abs(y/frame.shape[0] - 0.5)) > 0.25):
                continue

            # Fit the moving Gaussian function
            fit_params = fitMovingGaussian(frame, x, y)


            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)


            # Plot the original image
            ax1.imshow(frame, cmap='gray', vmin=0, vmax=saturation_level)
            ax1.scatter(x, y, c='r')
            ax1.set_title('Image')


            # test
            # a0, level_sum, sigma, L, omega
            #fit_params = [  0.0, 1.3250842e+04, 1.5, x, y, 10.0, 2.12906464e+00]

            # Generate the fitted Gaussian
            y_ind, x_ind = np.indices(frame.shape)
            img_fit = movingGaussian2D((y_ind, x_ind), *fit_params, saturation_level=saturation_level)
            img_fit = img_fit.reshape(frame.shape[1], frame.shape[0])

            # Plot the fit
            ax2.imshow(img_fit, cmap='gray', vmin=0, vmax=saturation_level)
            ax2.set_title('Fit')


            # Plot the residuals
            res = frame - img_fit
            max_dev = max(abs(np.min(res)), abs(np.max(res)))
            ax3.imshow(res, cmap='bwr', vmin=-max_dev, vmax=+max_dev)
            ax3.set_title('Residuals')

            print('Residual sum:', np.sum(res))
            print('Fitted flux:', fit_params[1])

            # Add the fitted and the image sum flux to the list
            flux_fit_list.append(fit_params[1])
            flux_sum_list.append(np.sum(frame))

            plt.show()

            #break


        # Plot the fitted flux curve
        plt.plot(flux_fit_list, label='Fitted')
        plt.plot(flux_sum_list, label='Summed')

        plt.ylabel('Flux')

        plt.legend()

        plt.show()
