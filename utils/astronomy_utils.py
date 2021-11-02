# ====================================================================================
# Dependencies
# ====================================================================================

import numpy as np
import math as m
import pandas as pd
from astropy import units as u
from astropy.stats import poisson_conf_interval
from scipy import stats
from tqdm import tqdm
from scipy.stats import norm


# ====================================================================================
# Euclidean-normalized counts
# ====================================================================================

def euclidean_counts(data, flux: str, s_range: tuple, N, area):
    """
    CALCULATES THE EUCLIDEAN-NORMALIZED FLUX COUNTS FOR A SET OF SOURCES

    :param data: Input data file
    :param flux: String for flux column
    :param s_range: Tuple for minimum and maximum flux [mJy]
    :param N: Number of data points required
    :param area: Survey area [square degrees]
    :return: Dictionary with flux, euclidean counts and low and high error bars
    """

    # Define the minimum and maximum flux range, convert to Jy and put in log space
    min_s, max_s = s_range
    min_s, max_s = min_s / 1000, max_s / 1000
    min_log_s, max_log_s = np.log10(min_s), np.log10(max_s)

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
    dn_domega_error_low = dn_errlow / (area.to(u.sr))
    dn_domega_error_high = dn_errhi / (area.to(u.sr))

    dn_domega_ds_error_low = dn_domega_error_low / delta_s
    dn_domega_ds_error_high = dn_domega_error_high / delta_s

    s025_dn_domega_ds_error_low = dn_domega_ds_error_low * s025
    s025_dn_domega_ds_error_high = dn_domega_ds_error_high * s025

    # This lines follows from the fact that the logarithm (base 10) of an error, del(x), is 0.434*del(x)/x
    ntotal_error_low = 0.434 * (s025_dn_domega_ds_error_low / s025_dn_domega_ds)
    ntotal_error_high = 0.434 * (s025_dn_domega_ds_error_high / s025_dn_domega_ds)

    # Centre the flux range
    s_range_centre = [(s_range[i] + s_range[i + 1]) / 2. for i in range(len(s_range) - 1)]
    s_range_mJy = [i * 1000 for i in s_range_centre]

    return {'flux': s_range_mJy, 'euclidean_counts': ntotal, 'error_low': ntotal_error_low,
            'error_high': ntotal_error_high}


# ====================================================================================
# Redshift distribution function
# ====================================================================================

def dn_dz_domega(data, min_z, max_z, n, area):
    """
    RETURNS THE REDSHIFT DISTRIBUTION IN UNITS OF d2N/dz/domega [dz-1 sr-1]

    :param data: Input data file
    :param min_z: Minimum redshift value
    :param max_z: Maximum redshift value
    :param n: Number of required bins
    :param area: Survey area [square degrees]
    :return: counts, bin_centres
    """

    # convert the area to steradians
    area = area * (u.deg ** 2)
    area = area.to(u.sr)

    # Generate a histogram for the data
    bins = np.linspace(min_z, max_z, n)
    area = np.full(n, area)
    counts, bin_edges = np.histogram(data, bins=bins)

    # Determine the counts in units of [dz-1 sr-1]
    bin_diff = np.diff(bin_edges)
    counts = [i / (j * k) for i, j, k in zip(counts, bin_diff, area)]
    bin_centres = [(bin_edges[i] + bin_edges[i + 1]) / 2. for i in range(len(bin_edges) - 1)]
    return counts, bin_centres


# ====================================================================================
# Lensing Probabilities inspired by Gonzalez-Nuevo et al, 2019
# ====================================================================================

def lensing_probabilities(data, redshift_source: str, redshift_lens: str, redshift_error_source: str,
                          redshift_error_lens: str):
    """
    CALCULATES THE LENSING PROBABILITIES OF A SET OF SOURCES

    :param data: Input data file
    :param redshift_source: String for redshift of the expected source column
    :param redshift_lens: String for redshift of the expected lens column
    :param redshift_error_source: String for redshift error of the expected source column
    :param redshift_error_lens: String for redshift error of the expected lens column
    :return: bc_list
    """

    # Function defining the Bhattacharyya coefficient (BC), a measure of the difference in probability distributions
    def BC(mu_p, mu_q, sigma_p, sigma_q):
        """
        CALCULATES THE BHATTACHARYYA COEFFICIENT

        :param mu_p: Mean value of the p Gaussian distribution
        :param mu_q: Mean value of the q Gaussian distribution
        :param sigma_p: Standard deviation of the p Gaussian distribution
        :param sigma_q: Standard deviation of the q Gaussian distribution
        :return: bc
        """
        d = (0.25 * np.log(0.25 * ((sigma_p ** 2 / sigma_q ** 2) + (sigma_q ** 2 / sigma_p ** 2) + 2))) + (
                0.25 * (((mu_p - mu_q) ** 2) / (sigma_p ** 2 + sigma_q ** 2)))
        bc = np.exp(-d)
        return bc

    # Calculate the BC factor for all sources
    bc_list = []
    for obj in tqdm(range(len(data)), desc='Calculating Lensing Probabilities'):
        if m.isnan(data[redshift_source][obj]) | m.isnan(data[redshift_lens][obj]) | m.isnan(
                data[redshift_error_source][obj]) | m.isnan(data[redshift_error_lens][obj]):
            bc = m.nan
            bc_list.append(bc)

        else:
            # Conventional BC factor
            bc = BC(data[redshift_source][obj], data[redshift_lens][obj], data[redshift_error_source][obj],
                    data[redshift_error_lens][obj])

            # Overlap factor
            mean = data[redshift_source][obj]
            std = data[redshift_error_source][obj]
            x = data[redshift_lens][obj] - (3 * data[redshift_error_lens][obj])
            bc_overlap = norm(mean, std).cdf(x)

            # Calculate the total BC value
            bc_combined = 1 - (bc + bc_overlap)

            # The overlap can sometimes leave a negative lensing probability
            if bc_combined < 0:
                bc_list.append(0)
            else:
                bc_list.append(bc_combined)

    return bc_list


# ====================================================================================
# Function to determine the optimal minimum lensing probability
# ====================================================================================

def optimal_lens_probability(data, reliability: str, lensing_probability: str, z_source: str, false_id_percent,
                             reliability_thresh=0.8, minimum_z_source=2.5, n=100):
    """
    FUNCTION DETERMINES THE OPTIMAL LENSING PROBABILITY THAT HAS THE LOWEST FALSE ID RATE

    :param data: Input data file
    :param reliability: String for the reliability column
    :param lensing_probability: String for the lensing probability column
    :param z_source: String for the redshift of the source column
    :param false_id_percent: Percentage false ID rate of the supplied data
    :param reliability_thresh: Minimum reliability threshold used to calculate lensing probabilities
    :param minimum_z_source: Minimum redshift of the source to always be considered lensed
    :param n: The number of optimal p values tested in the range [0, 1]
    :return: p_critical_range, p_false_positive, estimate1, estimate2, p_optimal
    """

    # Select only sources that meet our reliability threshold
    source_reliable_id = data[data[reliability] >= reliability_thresh]

    # Function defines the false positive rate for a given lensing probability
    def p_false(p_critical):
        lenses = source_reliable_id[(source_reliable_id[lensing_probability] >= p_critical)]
        n_lenses = len(lenses)
        lenses = lenses.reset_index()

        # First estimate: the sum of 1 - lensing probability of all sources
        n_unlensed_1 = 0
        for obj in range(n_lenses):
            n_unlensed_value = 1 - lenses[lensing_probability][obj]
            n_unlensed_1 += n_unlensed_value

        # Second estimate: number of sources with high z multiplied by the fraction of sources likely to be spurious
        n_unlensed_2 = len(source_reliable_id) * (false_id_percent / 100) * (
                (len(source_reliable_id[source_reliable_id[z_source] > minimum_z_source])) / (
            len(source_reliable_id)))

        return n_unlensed_1 / n_lenses, n_unlensed_2 / n_lenses, (n_unlensed_1 + n_unlensed_2) / n_lenses

    # Calculating the false positive rate for a range of critical lensing probabilities
    p_critical_range = np.linspace(0.01, 0.99, n)

    # Estimating the false positive rate (including estimate 1 and estimate 2 methods)
    estimate1 = []
    estimate2 = []
    p_false_positive = []
    for p in tqdm(p_critical_range, desc='Calculating the False Positive rates'):
        estimate1_value, estimate2_value, false_positive = p_false(p)
        p_false_positive.append(false_positive)
        estimate1.append(estimate1_value)
        estimate2.append(estimate2_value)

    # Determining the optimal minimum lensing probability
    index_min = np.argmin(p_false_positive)
    p_optimal = p_critical_range[index_min]

    return p_critical_range, p_false_positive, estimate1, estimate2, p_optimal


# =========================================================================================================
# Function to clean a sample of potential lensing candidates of local galaxies, blazars and variable stars
# =========================================================================================================

def clean_lensed_candidates(data, f250: str, f350: str, identification: str, var_stars: list, blazars: list):
    """
    FUNCTION CLEANS A SAMPLE OF LOCAL GALAXIES, BLAZARS AND VARIABLE STARS

    :param data: Input data file
    :param f250: String for the 250-micron flux column [Jy]
    :param f350: String for the 350-micron flux column [Jy]
    :param identification: String for the ID column for which variable stars and blazars are known
    :param var_stars: List of variable stars IDs for removal
    :param blazars: List of blazar IDs for removal
    :return: candidates, local_galaxies, all_var_stars, all_blazars
    """

    # Restrict ourselves to distant objects to remove local galaxies
    candidates = data[(data[f250] / data[f350] < 1.5)]
    local_galaxies = data[(data[f250] / data[f350] >= 1.5)]

    # Remove variable stars from list
    for star in var_stars:
        star_found = data[data[identification] == star]
        candidates = pd.concat([candidates, star_found, star_found]).drop_duplicates(keep=False)

    # Remove blazars from list
    for blazar in blazars:
        blazar_found = data[data[identification] == blazar]
        candidates = pd.concat([candidates, blazar_found, blazar_found]).drop_duplicates(keep=False)

    # Collect all variable stars and blazars
    all_var_stars = data.loc[data[identification].isin(var_stars)]
    all_blazars = data.loc[data[identification].isin(blazars)]

    return candidates, local_galaxies, all_var_stars, all_blazars


# =========================================================================================================
# Function that splits the lensed sample from the full sample
# =========================================================================================================

def lens_split(data, f500: str, reliability: str, lens_probability: str, lens_probability_thresh, reliability_thresh=0.8):
    """
    RETURNS THE LENSED CANDIDATES FROM A SET OF SOURCES

    :param data: Input data file
    :param f500: String for the 500-micron flux column [Jy]
    :param reliability: String for the reliability column
    :param lens_probability: String for the lensing probability column
    :param lens_probability_thresh: The minimum lensing probability for < 100 mJy candidates
    :param reliability_thresh: The minimum reliability threshold for < 100 mJy candidates
    :return: candidates
    """

    # Conditions for the <100 mJy sample to be lensed
    candidates = data[(data[f500] < 0.1) &
                      (data[reliability] > reliability_thresh) &
                      (data[lens_probability] >= lens_probability_thresh)]

    return candidates


# =========================================================================================================
# Cumulative number counts function
# =========================================================================================================

def cumulative_counts(data, flux: str, s_range: tuple, N, area):
    """
    CALCULATES THE CUMULATIVE COUNTS FOR A SET OF SOURCES

    :param data: Input data file
    :param flux: String for the flux column [Jy]
    :param s_range: Tuple defining the minimum and maximum flux values [mJy]
    :param N: Number of data points required
    :param area: Survey area [square degrees]
    :return: s_range_mJy, n_gts, n_gts_error_low, n_gts_error_high
    """

    # Define the minimum and maximum flux range, convert to Jy and put in log space
    min_s, max_s = s_range
    min_s, max_s = min_s / 1000, max_s / 1000
    min_log_s, max_log_s = np.log10(min_s), np.log10(max_s)
    s_range_Jy = np.sort(np.logspace(min_log_s, max_log_s, N))
    s_range_mJy = [s * 1000 for s in s_range_Jy]

    # Calculate the number greater than a given flux value
    n_gts = []
    n = []

    for s in s_range_Jy:
        n_value = len([flux for flux in data[flux] if flux >= s])
        n.append(n_value)
        n_gts.append(n_value / area)

    # Define the confidence interval based on Poisson counting statistics
    n_abserr_low, n_abserr_high = poisson_conf_interval(n)

    # calculate the upper and lower limits on our counts
    n_error_low = [i - j for i, j in zip(n, n_abserr_low)]
    n_error_high = [j - i for i, j in zip(n, n_abserr_high)]
    n_gts_error_low = [(i / area) for i in n_error_low]
    n_gts_error_high = [(i / area) for i in n_error_high]

    return s_range_mJy, n_gts, n_gts_error_low, n_gts_error_high


# =========================================================================================================
# Lensing fraction function
# =========================================================================================================

def lensing_fraction(data, lensed_candidates, f500: str, s_range: tuple, N, area):
    """
    RETURNS THE LENSING FRACTION GIVEN A SAMPLE OF CANDIDATE LENSES AND THE TOTAL SAMPLE

    :param data: Input data file
    :param lensed_candidates: Input lensed candidates table
    :param f500: String for the 500-micron flux column [Jy]
    :param s_range: Tuple defining the minimum and maximum flux values [mJy]
    :param N: Number of data points required
    :param area: Survey area [square degrees]
    :return: s_range_mJy, frac500
    """

    # Define the minimum and maximum flux range, convert to Jy and put in log space
    min_s, max_s = s_range
    min_s, max_s = min_s / 1000, max_s / 1000
    min_log_s, max_log_s = np.log10(min_s), np.log10(max_s)
    s_range = np.logspace(min_log_s, max_log_s, N)
    s_range_mJy = [i * 1000 for i in s_range]

    # Calculate the number greater than a given flux value
    n_gts_total = []
    n_gts_lens = []

    for s in s_range:
        n_total_value = len([flux for flux in data[f500] if flux >= s])
        n_gts_total.append(n_total_value / area)
        n_lens_value = len([flux for flux in lensed_candidates[f500] if flux >= s])
        n_gts_lens.append(n_lens_value / area)

    # Calculating the lensing fraction
    frac500 = [lens / total for lens, total in zip(n_gts_lens, n_gts_total)]

    return s_range_mJy, frac500


# =========================================================================================================
# Genuine multiple counterparts function
# =========================================================================================================

def genuine_multiples(data, distance: str, groupid: str, redshift: str, redshift_errors: str, maximum_radius=8):
    """
    FUNCTION DETERMINES THE DISTRIBUTION OF delta(z)/sigma(delta(z)) USED FOR THE DETECTION OF MULTIPLES

    :param data: Input data file
    :param distance: String for separation column
    :param groupid: String for the group's ID
    :param redshift: String for the counterpart's redshift column
    :param redshift_errors: String for the counterpart's redshift errors column
    :param maximum_radius: The maximum radius to search for multiple genuine counterparts [arcsec]
    :return: delta_div_sigma
    """

    # Limit the data to the radius where multiples are likely to occur
    data_limit = data[data[distance] <= maximum_radius]

    # Function to find the closest pair in an array, which will be used to find the closest redshift pair
    def closest_pair(arr, n):
        b = list(arr)
        if n <= 1: return
        arr.sort()
        minDiff = arr[1] - arr[0]

        for i in range(2, n):
            minDiff = min(minDiff, arr[i] - arr[i - 1])

        for i in range(1, n):
            if (arr[i] - arr[i - 1]) == minDiff:
                if b.index(arr[i]) > b.index(arr[i - 1]):
                    return arr[i - 1], arr[i]
                else:
                    return arr[i], arr[i - 1]

    # We divide the data into their ID groups
    data_groups = data_limit.groupby(groupid)

    z_pairs = []
    z_error_pairs = []
    # Finding closest pairs and appending the redshifts and errors to lists
    for name, group in tqdm(data_groups, desc="Finding Pairs"):

        # We randomize the group so that delta(z) can be positive or negative, and then find the closest pair
        group = group.sample(frac=1, replace=False)
        pair = closest_pair(np.array(group[redshift]), len(group[redshift]))

        # If we find a pair
        if pair is not None:

            # Obtain their redshifts
            z1 = pair[0]
            z2 = pair[1]
            z_pairs.append(pair)

            # Obtain their corresponding redshift errors
            z1_error = group.loc[group[redshift] == z1, redshift_errors].iloc[0]
            z2_error = group.loc[group[redshift] == z2, redshift_errors].iloc[0]
            z_error_pairs.append((z1_error, z2_error))

        # If we do not find a pair
        else:
            z_pairs.append((np.nan, np.nan))
            z_error_pairs.append((np.nan, np.nan))

    # Calculating delta(z)/sigma(delta(z))
    delta_div_sigma = []
    for pair in range(len(z_pairs)):
        delta_z = z_pairs[pair][1] - z_pairs[pair][0]
        sigma_delta_z = np.sqrt((z_error_pairs[pair][1] ** 2) + (z_error_pairs[pair][0] ** 2))
        delta_z_div_sigma_delta_z = delta_z / sigma_delta_z
        delta_div_sigma.append(delta_z_div_sigma_delta_z)

    return delta_div_sigma


