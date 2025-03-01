import numpy as np
import joblib
import os
import pickle
import sys

# GMM Prior
"""gmm_prior_dir = './prior_base/fire_batch_gmm_prior'

for event in os.listdir(gmm_prior_dir):
    ctr_path = gmm_prior_dir + '/' + event + '/prior_file_ctr.npy'
    arr = np.load(ctr_path)
    ratio = 4096. / arr.shape[0]
    size = 4096 / ratio
    print(event, size, ratio)"""

# AGG Prior
"""agg_prior_dir = './fire_batch_agg_prior'
for event in os.listdir(agg_prior_dir):
    ctr_path = gmm_prior_dir + '/' + event + '/prior_file_ctr.npy'
    arr = np.load(ctr_path)
    ratio = 4096. / arr.shape[0]
    size = 4096 / ratio
    print(event, size, ratio)"""

# IFR Prior
def extract_tree_info(tree):
    tree_info = {
        'children_left': tree.tree_.children_left,
        'children_right': tree.tree_.children_right,
        'threshold': tree.tree_.threshold,
        'feature': tree.tree_.feature,
    }
    return tree_info

ifr_prior_dir = './fire_batch_ifr_prior'
for event in os.listdir(ifr_prior_dir):
    pri_path = ifr_prior_dir + '/' + event+ '/isolation_forest_model.pkl'
    ifr_model = joblib.load(pri_path)

    minimal_trees = [extract_tree_info(tree) for tree in ifr_model.estimators_]
    serialized_minimal_trees = pickle.dumps(minimal_trees)
    minimal_trees_size_bytes = sys.getsizeof(serialized_minimal_trees)

    # assume 32 bit (/2.) and use KB for plot (/1024.), therefore use /2048.
    space = minimal_trees_size_bytes/2048.
    ratio = 4096./space
    print(event, space, ratio)
