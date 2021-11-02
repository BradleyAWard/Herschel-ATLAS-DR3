# ====================================================================================
# Dependencies
# ====================================================================================

import os
import pandas as pd
import numpy as np
from astropy.table import Table
from tqdm import tqdm


# ====================================================================================
# Changes data to a different search radius (Must be smaller than currently used)
# ====================================================================================

def change_radius(file_name: str, sourceid: str, groupid: str, groupsize: str, distance: str, additional_cols: int,
                  r=10):
    """ CHANGES THE SEARCH RADIUS FOR A CATALOGUE

    :param file_name: String for the input file (with r greater than you wish to change to)
    :param sourceid: String for the identifier column for multi-groups
    :param groupid: String for the group ID column
    :param groupsize: String for the group size column
    :param distance: String for the distance (angular separation) column [arcsec]
    :param additional_cols: Integer for the column beyond which is cross-matched data
    :param r: Float for the new search radius of the output catalogue
    :return: table
    """

    # Load in the original data
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    print(ROOT)
    file_path = ROOT + '/data/clean_dr3_files/' + file_name
    input_data = Table.read(file_path).to_pandas()

    # Separate the original data into blanks, single objects and multiple objects
    blanks = input_data[input_data[groupid].isnull() & (input_data[distance] == 0)]
    blanks = blanks.reset_index()
    blanks_format = blanks.iloc[0, additional_cols:]

    singles = input_data[input_data[groupid].isnull() & (input_data[distance] != 0)]
    singles = singles.reset_index()

    multi = input_data[input_data[groupid].isnull() == False]
    multi = multi.reset_index()

    # Converting original blanks
    blanks_from_blanks = blanks

    # Converting original singles
    blanks_from_singles = []
    singles_from_singles = []
    for obj in tqdm(range(len(singles)), desc='Converting Single objects'):
        if singles[distance][obj] <= r:
            singles_from_singles.append(singles.iloc[obj])
        else:
            original = singles.iloc[obj, :additional_cols]
            new = blanks_format
            new_blank = pd.concat([original, new])
            blanks_from_singles.append(new_blank)

    # Converting original multiples
    blanks_from_multi = []
    singles_from_multi = []
    multi_from_multi = []
    test1 = multi.groupby(sourceid)
    for name, group in tqdm(test1, desc='Converting Multiple Objects'):
        new = group[group[distance] <= r]
        new[groupsize] = len(new)

        if len(new) == 1:
            new[groupsize] = np.nan
            new[groupid] = np.nan
            singles_from_multi.append(new)

        elif len(new) == 0:
            original = group.iloc[0, :additional_cols]
            new_blank_format = blanks_format
            new = pd.concat([original, new_blank_format])
            blanks_from_multi.append(new)

        else:
            multi_from_multi.append(new)

    # Converting all new tables into dataframes
    print('Finalizing the final table, this may take a couple of minutes')
    blanks_from_singles = pd.DataFrame(blanks_from_singles)
    singles_from_singles = pd.DataFrame(singles_from_singles)
    blanks_from_multi = pd.DataFrame(blanks_from_multi)
    singles_from_multi = pd.concat(singles_from_multi)
    multi_from_multi = pd.concat(multi_from_multi)
    table = pd.concat([blanks_from_blanks, blanks_from_singles, singles_from_singles,
                       blanks_from_multi, singles_from_multi, multi_from_multi])

    output_path = ROOT + '/data/' + file_name + '_' + '{}'.format(r)
    t_table = Table.from_pandas(table)
    t_table.write(output_path, overwrite=True, format='csv')


