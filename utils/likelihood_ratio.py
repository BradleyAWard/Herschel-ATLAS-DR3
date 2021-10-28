# ====================================================================================
# Dependencies
# ====================================================================================

import numpy as np
import math as m
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm


# ====================================================================================
# Function for producing n distributions
# ====================================================================================

def N(data, background, Ndata, Nback, r, band: str, bins=40, mag_min=9, mag_max=22):
    """ FUNCTION THAT CREATES n(real) and n(normalized) DISTRIBUTIONS FOR A SET OF SOURCES """

    # Create histograms for the data, background and n(real)
    n_data, mag = np.histogram(data[band], bins=bins, range=(mag_min, mag_max))
    n_back, _ = np.histogram(background[band], bins=bins, range=(mag_min, mag_max))
    n_real = n_data - n_back * (Ndata / Nback)

    # Create normalized versions of the n distributions
    n_data_normalized = n_data / (Ndata * np.pi * (r ** 2))
    n_back_normalized = n_back / (Nback * np.pi * (r ** 2))

    # Define the bin centres for plotting
    bin_centres = [(mag[i] + mag[i + 1]) / 2. for i in range(len(mag) - 1)]

    return bin_centres, n_real, n_data_normalized, n_back_normalized


# ====================================================================================
# Function for producing q/n distributions
# ====================================================================================

def q_div_n(n_real, n_normalized, q0, bincentres, window_size, mag_min=9, mag_max=22, n=100):
    """ RETURNS THE q/n DISTRIBUTION GIVEN n(real) """

    q = q0 * (n_real / sum(n_real))
    q_over_n = q / n_normalized

    # We drop any values in q/n which are negative, NaN or infinite so that we can interpolate the function
    qn_cut = [qn for qn, bins in zip(q_over_n, bincentres) if
              (qn > 0) & (m.isnan(qn) == False) & (m.isinf(qn) == False)]
    bins_cut = [bins for qn, bins in zip(q_over_n, bincentres) if
                (qn > 0) & (m.isnan(qn) == False) & (m.isinf(qn) == False)]

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    # Calculate the moving average of q/n
    averaged_qn = moving_average(qn_cut, window_size)
    averaged_mag = moving_average(bins_cut, window_size)

    # Interpolate over all complete values of q/n
    mag_range = np.linspace(mag_min, mag_max, n)
    interp_qn_function = interp1d(averaged_mag, averaged_qn, kind='linear', bounds_error=False)
    interpolated_qn = interp_qn_function(mag_range)

    # Replace NaN values with the nearest true value
    # TODO - Check that replacement of NaNs is correct
    ind = np.where(~np.isnan(interpolated_qn))[0]
    first, last = ind[0], ind[-1]
    interpolated_qn[:first] = interpolated_qn[first]
    interpolated_qn[last + 1:] = interpolated_qn[last]

    return mag_range, q, q_over_n, interpolated_qn


# ====================================================================================
# Function for calculating likelihood values
# ====================================================================================

def likelihood(data, counterpart_id: str, f250: str, e250: str, SG: str, distance: str, band: str, k, fwhm,
               qn_gal: tuple, qn_stars: tuple):
    """ RETURNS THE LR VALUES FOR A SET OF SOURCES """

    likelihood_ratios = []
    for obj in tqdm(range(len(data)), desc='Likelihood Ratio calculations'):

        # Calculate the snr, individual sigma(pos) and f for each source
        snr = data[f250][obj] / data[e250][obj]
        sigma = (k * fwhm) / snr
        f = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(data[distance][obj] ** 2) / (2 * (sigma ** 2)))

        # If there is no counterpart do not add a likelihood value
        if pd.isnull(data[counterpart_id][obj]):
            likelihood_ratios.append(np.nan)
            continue

        # Collect the q/n distribution for either a galaxy or star
        if data[SG][obj] == 1:
            mag_range, qn = qn_gal
        else:
            mag_range, qn = qn_stars

        def find_nearest(x_val, x, y):
            """ FIND THE VALUE FROM y THAT CORRESPONDS TO THE CLOSEST VALUE IN x """
            difference = lambda list_val: abs(list_val - x_val)
            closest_x = min(x, key=difference)
            index = list(x).index(closest_x)
            nearest_y = y[index]
            return nearest_y

        # Find the nearest q/n value and calculate the likelihood ratio
        object_magnitude = data[band][obj]
        qn_val = find_nearest(object_magnitude, mag_range, qn)
        lr = f * qn_val
        likelihood_ratios.append(lr)

    return likelihood_ratios


# ====================================================================================
# Function for calculating reliability values
# ====================================================================================

def reliability(data, counterpart_id: str, groupid: str, likelihood: str, q0):
    """ RETURNS THE RELIABILITY VALUES FOR A SET OF SOURCES """
    sum_likelihood = data.groupby([groupid])[likelihood]
    data = data.assign(sum_lr=sum_likelihood.transform(sum))

    r = []
    for obj in tqdm(range(len(data)), desc='Reliability calculations'):

        # If there is no counterpart do not add a reliability value
        if pd.isnull(data[counterpart_id][obj]):
            r.append(np.nan)
            continue

        # If single object
        if m.isnan(data[groupid][obj]):
            r_val = (data[likelihood][obj] / (data[likelihood][obj] + 1))
            r.append(r_val)

        # If multiple objects
        else:
            r_val = data[likelihood][obj] / (data['sum_lr'][obj] + (1 - q0))
            r.append(r_val)

    return r
