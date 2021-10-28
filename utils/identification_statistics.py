# ====================================================================================
# Dependencies
# ====================================================================================

import numpy as np
import pandas as pd
from prettytable import PrettyTable

# ====================================================================================
# False identification rate function
# ====================================================================================

def N_false(data, reliability: str, r_thresh):
    """ FUNCTION TO DETERMINE THE FALSE ID RATE OF A SET OF SOURCES """
    false_ids = [(1 - r) for r in data[reliability] if r >= r_thresh]
    n_false_ids = np.sum(false_ids)
    n_false_ids_percent = n_false_ids / len(data) * 100
    return n_false_ids, n_false_ids_percent


# ====================================================================================
# Completeness function
# ====================================================================================

def completeness(data, reliability: str, f250: str, e250: str, q0, r_thresh, snr_thresh=4):
    """ FUNCTION TO CALCULATE THE COMPLETENESS OF A SET OF SOURCES """
    n_reliable = [rel for rel in data[reliability] if rel >= r_thresh]
    n_snr = [snr for snr in (data[f250] / data[e250]) if snr >= snr_thresh]
    eta = len(n_reliable) / (len(n_snr) * q0)
    return eta


# ====================================================================================
# Cleanness function
# ====================================================================================

def cleanness(data, reliability: str, r_thresh):
    """ FUNCTION TO CALCULATE THE CLEANNESS OF A SET OF SOURCES """
    false_ids = [(1 - r) for r in data[reliability] if r >= r_thresh]
    n_false_ids = np.sum(false_ids)
    c = 1 - (n_false_ids / len(data))
    return c


# ====================================================================================
# Colours function
# ====================================================================================

def colour_split(data, f250: str, f350: str, red_green, green_blue):
    """ FUNCTION THAT SPLITS A SET OF SOURCES INTO THREE COLOURS """
    blue = data[(data[f250] / data[f350]) > green_blue]
    green = data[((data[f250] / data[f350]) > red_green) & ((data[f250] / data[f350]) < green_blue)]
    red = data[(data[f250] / data[f350]) < red_green]
    return red, green, blue


# ====================================================================================
# Function for reliable percentage as a function of multiplicity
# ====================================================================================

def multiplicity_reliability(sources, groupid: str, groupsize: str, distance: str, reliability: str, r_thresh=0.8, max_counterparts=10):
    """ FUNCTION GENERATES A TABLE FOR THE RELIABLE PERCENTAGE OF IDs AS A FUNCTION OF THE NUMBER OF IDs FOUND """

    # Setup a table for the output
    t = PrettyTable(['N (Match)', 'N (Data)', 'N (Reliable)', 'Reliable Percentage', 'Average Separation'], float_format=".2")

    # Determine the number of sources with 0 and 1 counterparts, and calculate how many are reliable
    zero_sources = len(sources[(pd.isnull(sources[groupid])) & (sources[distance] == 0)])
    one_sources = len(sources[(pd.isnull(sources[groupid])) & (sources[distance] != 0)])
    zero_reliable_sources = len(sources[(pd.isnull(sources[groupid])) & (sources[distance] == 0) & (sources[reliability] > r_thresh)])
    one_reliable_sources = len(sources[(pd.isnull(sources[groupid])) & (sources[distance] != 0) & (sources[reliability] > r_thresh)])
    zero_percentage = (zero_reliable_sources/zero_sources)*100
    one_percentage = (one_reliable_sources/one_sources) * 100

    # Determine the average separation distance for sources with 0 and 1 counterparts
    zero_distance = sources[(pd.isnull(sources[groupid])) & (sources[distance] == 0)][distance].mean()
    one_distance = sources[(pd.isnull(sources[groupid])) & (sources[distance] != 0)][distance].mean()

    # Add the rows for 0 and 1 counterparts to the table
    t.add_row([0, zero_sources, zero_reliable_sources, zero_percentage, zero_distance])
    t.add_row([1, one_sources, one_reliable_sources, one_percentage, one_distance])

    # For two or more counterparts, calculate the number that are reliable, the average separation and add to the table
    for obj_number in range(2, max_counterparts+1):
        n_sources = len(sources[sources[groupsize] == obj_number])
        n_sources_reliable = len(sources[(sources[groupsize] == obj_number) & (sources[reliability] > r_thresh)])
        reliable_percentage = (n_sources_reliable/n_sources)*100
        average_separation = sources[sources[groupsize] == obj_number][distance].mean()
        t.add_row([obj_number, n_sources, n_sources_reliable, reliable_percentage, average_separation])

    return t


