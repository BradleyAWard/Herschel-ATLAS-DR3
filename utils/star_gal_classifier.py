# ====================================================================================
# Dependencies
# ====================================================================================

import pandas as pd
import numpy as np
from tqdm import tqdm

# ====================================================================================
# Baldry et al, 2010 stellar locus converted to Vega system
# ====================================================================================

def stellar_locus(x, offset):
    """
    BALDRY ET AL, 2010 STELLAR LOCUS FUNCTION

    :param x: Value of g-i colour
    :param offset: Offset from stellar locus for star-galaxy classifier
    """

    if x < 0.3:
        return 0.2228 + offset
    if (x >= 0.3) & (x < 2.3):
        return 0.05 + 0.615*x - 0.13*(x**2) + offset
    if x >= 2.3:
        return 0.7768 + offset

# ====================================================================================
# Function for star-galaxy classification
# ====================================================================================

def classification(data, counterpart_id: str, j: str, k: str, g: str, i: str, pstar: str):
    """
    FUNCTION THAT CLASSIFIES A DATABASE WITH J-K AND g-i COLOURS

    :param data: Input data file
    :param counterpart_id: String for the counterpart IDs column
    :param j: String for the J magnitude column
    :param K: String for the K magnitude column
    :param g: String for the g magnitude column
    :param i: String for the i magnitude column
    :param pstar: String for the VIKING stellar probability column
    :return: classes
    """

    jk = data[j] - data[k]
    # TODO - gi variable needs to be changed back to g - i when the data is downloaded
    gi = data[g] - data[i]
    jk_cut = [stellar_locus(colour, 0.2) for colour in gi]

    classes = []
    for obj in tqdm(range(len(data)), desc = 'Star-Galaxy Classification'):

        # If there is no counterpart do not give it a classification
        if pd.isnull(data[counterpart_id][obj]):
            classes.append(np.nan)
            continue

        # If P(star) > 95%, classify as a star
        elif data[pstar][obj] >= 0.95:
            classes.append(0)

        # If there is no g-i value to generate a J-K cut value
        elif (jk_cut[obj] == None):
            # If J-K is larger than the RHS of the stellar locus, classify as galaxy
            if (jk[obj] > stellar_locus(1000, 0.2)) & (jk[obj] < 1000):
                classes.append(3)
                continue
            # If J-K is less than the LHS of the stellar locus, classify as a star
            elif (jk[obj] < stellar_locus(0, 0.2)) & (jk[obj] > 0.0001):
                classes.append(4)
                continue
            # If P(star) > 70%, classify as a star
            elif data[pstar][obj] >= 0.7:
                classes.append(5)
                continue
            # Otherwise classify as a galaxy
            else:
                classes.append(6)
                continue

        # If there is a g-i value to generate a J-K cut value
        # If J-K is above the cut value and within bounds, classify as a galaxy
        elif (jk[obj] > jk_cut[obj]) & (jk[obj] > -6) & (jk[obj] < 6) & (gi[obj] > -6) & (gi[obj] < 6):
            classes.append(1)
        # If J-K is below the cut value and within bounds, classify as a star
        elif (jk[obj] < jk_cut[obj]) & (jk[obj] > -6) & (jk[obj] < 6) & (gi[obj] > -6) & (gi[obj] < 6):
            classes.append(2)
        # If J-K is larger than the RHS of the stellar locus, classify as galaxy
        elif (jk[obj] > stellar_locus(1000, 0.2)) & (jk[obj] < 1000):
            classes.append(3)
        # If J-K is less than the LHS of the stellar locus, classify as a star
        elif (jk[obj] < stellar_locus(0, 0.2)) & (jk[obj] > 0.0001):
            classes.append(4)
        # If P(star) > 70%, classify as a star
        elif data[pstar][obj] >= 0.7:
            classes.append(5)
        # Otherwise classify as a galaxy
        else:
            classes.append(6)

    return classes
