# ======================================================================================
# Dependencies
# ======================================================================================

from utils.rcparams import rcparams
from utils.data_loader import full_loader
from utils.star_gal_classifier import stellar_locus, classification
from utils.constants import blanks, B, K
from utils.likelihood_ratio import N, q_div_n, likelihood, reliability