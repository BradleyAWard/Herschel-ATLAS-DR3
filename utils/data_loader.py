# ====================================================================================
# Dependencies
# =====================================================================================

import os
import numpy as np
from pandas import read_csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# ====================================================================================
# Full data loader
# =====================================================================================

def full_loader(file_name):
    """
    RETURNS THE FULL DATABASE

    :param file_name: Name of the data file located in 'data'
    :return: data
    """
    file_path = ROOT + '/data/' + file_name
    data = read_csv(file_path)

    return data

