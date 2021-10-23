# ====================================================================================
# Dependencies
# ====================================================================================

import numpy as np
from astropy import units as u
from astropy.stats import poisson_conf_interval
from scipy import stats


# ====================================================================================
# Euclidean-normalized counts
# ====================================================================================

def euclidean_counts(data, flux: str, s_range: tuple, N, area):
    """ CALCULATES THE EUCLIDEAN-NORMALIZED FLUX COUNTS FOR A SET OF SOURCES """

    # Define the minimum and maximum flux range, convert to Jy and put in log space
    min_s, max_s = s_range
    min_s, max_s = min_s / 1000, max_s / 1000
    min_log_s = np.log10(min_s)
    max_log_s = np.log10(max_s)

    # Adds a random jitter to the flux array for plotting purposes
    def rand_jitter(arr):
        std = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * std

    # Add random jitter to flux range and calculate delta(s)
    s_range = np.sort(rand_jitter(np.logspace(min_log_s, max_log_s, N)))
    area = area * (u.deg ** 2)
    delta_s = np.diff(s_range)

    # Calculate the Euclidean counts dN/dS S^2.5
    dn, _ = np.histogram(data[flux], bins=s_range)
    dn_domega = dn / (area.to(u.sr))
    dn_domega_ds = dn_domega / delta_s
    s025 = (stats.binned_statistic(data[flux], data[flux], bins=s_range)[0]) ** 2.5
    s025_dn_domega_ds = dn_domega_ds * s025
    ntotal = np.log10(s025_dn_domega_ds * u.sr)

    # Define the confidence interval based on Poisson counting statistics
    dn_abserrlow, dn_abserrhi = poisson_conf_interval(dn)
    dn_errlow = [i - j for i, j in zip(dn, dn_abserrlow)]
    dn_errhi = [j - i for i, j in zip(dn, dn_abserrhi)]

    # Follow the same procedure on the errors as the values above
    dn_domega_errlow = dn_errlow / (area.to(u.sr))
    dn_domega_errhi = dn_errhi / (area.to(u.sr))

    dn_domega_ds_errlow = dn_domega_errlow / delta_s
    dn_domega_ds_errhi = dn_domega_errhi / delta_s

    s025_dn_domega_ds_errlow = dn_domega_ds_errlow * s025
    s025_dn_domega_ds_errhi = dn_domega_ds_errhi * s025

    # This lines follows from the fact that the logarithm (base 10) of an error, del(x), is 0.434*del(x)/x
    ntotal_errlow = 0.434 * (s025_dn_domega_ds_errlow / s025_dn_domega_ds)
    ntotal_errhi = 0.434 * (s025_dn_domega_ds_errhi / s025_dn_domega_ds)

    # Centre the flux range
    s_range_centre = [(s_range[i] + s_range[i + 1]) / 2. for i in range(len(s_range) - 1)]
    s_range_mJy = [i * 1000 for i in s_range_centre]

    return {'flux': s_range_mJy, 'euclidean_counts': ntotal, 'error_low': ntotal_errlow, 'error_high': ntotal_errhi}
