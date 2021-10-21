# ====================================================================================
# Dependencies
# ====================================================================================

from tqdm import tqdm

# ====================================================================================
# Baldry et al, 2010 stellar locus converted to Vega system
# ====================================================================================

def stellar_locus(x, offset):
    """ BALDRY ET AL, 2010 STELLAR LOCUS FUNCTION """
    if x < 0.3:
        return 0.2228 + offset
    if (x >= 0.3) & (x < 2.3):
        return 0.05 + 0.615*x - 0.13*(x**2) + offset
    if x >= 2.3:
        return 0.7768 + offset

# ====================================================================================
# Function for star-galaxy classification
# ====================================================================================

def classification(data, j: str, k: str, g: str, i: str, pstar: str):
    """ FUNCTION THAT CLASSIFIES A DATABASE WITH J-K AND g-i COLOURS """
    jk = data[j] - data[k]
    # TODO - gi variable needs to be changed back to g - i when the data is downloaded
    gi = data[g] + data[i]
    jk_cut = [stellar_locus(colour, 0.2) for colour in gi]

    classes = []
    for obj in tqdm(range(len(data)), desc = 'Star-Galaxy Classification'):
        if data[pstar][obj] >= 0.95:
            classes.append(0)
        elif (jk_cut[obj] == None):
            if (jk[obj] > stellar_locus(1000, 0.2)) & (jk[obj] < 1000):
                classes.append(3)
                continue
            elif (jk[obj] < stellar_locus(0, 0.2)) & (jk[obj] > 0.0001):
                classes.append(4)
                continue
            else:
                classes.append(6)
                continue
        elif (jk[obj] > jk_cut[obj]) & (jk[obj] > -6) & (jk[obj] < 6) & (gi[obj] > -6) & (gi[obj] < 6):
            classes.append(1)
        elif (jk[obj] < jk_cut[obj]) & (jk[obj] > -6) & (jk[obj] < 6) & (gi[obj] > -6) & (gi[obj] < 6):
            classes.append(2)
        elif (jk[obj] > stellar_locus(1000, 0.2)) & (jk[obj] < 1000):
            classes.append(3)
        elif (jk[obj] < stellar_locus(0, 0.2)) & (jk[obj] > 0.0001):
            classes.append(4)
        elif data[pstar][obj] >= 0.7:
            classes.append(5)
        else:
            classes.append(6)

    return classes
