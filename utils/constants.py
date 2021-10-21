# ====================================================================================
# Dependencies
# ====================================================================================

import numpy as np
import math as m
from tqdm import tqdm


# ====================================================================================
# Function to calculate the value of Q
# ====================================================================================

def blanks(data, r_limit, groupid: str, distance: str):
    """ FUNCTION TO CALCULATE THE NUMBER OF BLANK POSITIONS FOR A GIVEN MAXIMUM RADIUS """
    smallest_r = data.groupby([groupid])[distance]
    data = data.assign(min_r=smallest_r.transform(min))

    blank0 = len([r for idx, r in zip(data[groupid], data[distance]) if (m.isnan(idx)) & (r == 0)])
    blank1 = len([r for idx, r in zip(data[groupid], data[distance]) if (m.isnan(idx)) & (r != 0) & (r > r_limit)])
    blankmulti = len(
        np.unique([idx for idx, r in zip(data[groupid], data['min_r']) if (m.isnan(idx) == False) & (r > r_limit)]))
    blank_total = blank0 + blank1 + blankmulti

    return blank_total


# ====================================================================================
# Expected functional form for the distribution of blanks
# ====================================================================================

def B(r, sigma, Q0):
    """ FUNCTION FOR THE DISTRIBUTION OF BLANKS """
    f = 1 - np.exp(-(r**2)/(2*(sigma**2)))
    return 1 - Q0*f


# ====================================================================================
# Function to calculate the expected k-constant
# ====================================================================================

def K(data, fwhm, sigma, f250: str, e250: str):
    """ FUNCTION TO CALCULATE A K-CONSTANT VALUE FROM ALL SOURCES """

    k_values = []
    for obj in tqdm(range(len(data)), desc='Calculating K-constant values'):
        snr = data[f250][obj]/data[e250][obj]
        k_values.append(sigma*snr/fwhm)

    return k_values
