from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
from toolkits import deep_features, construct_feature_extractor, read_images
from feature_filters import channel_mean_image
import os
import shutil
from tqdm import tqdm
from pyod.models.iforest import IForest
import joblib
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering

band_choices = {
    'optic': [3, 2, 1], 'fire': [5, 4, 2], 'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0], 'l7_veg': [3, 2, 1], 'l7_fire': [4, 3, 2],
    'band3': [2, 1, 0]
}
def compute_bic(n, ref_feat_dm):
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(ref_feat_dm)
    return gmm.bic(ref_feat_dm)

def ifr_prior(prior_image_path, predictor, save_dir, band_choice='optic'):
    # step-1: use gdal to fetch prior image
    prior_image = read_images(prior_image_path, band_choice=band_choices[band_choice], file_tpye='gdal')

    # step-2: use sam predictor to fetch characterization of prior image
    ref_feat = deep_features(prior_image, predictor, layer=-1)
    ref_feat = channel_mean_image(ref_feat, near_radius=2)

    # step-3 train an isolation forest
    rh, rw, rc = ref_feat.shape
    ref_feat_flat = ref_feat.reshape(rh * rw, rc)

    clf = IForest()
    clf.fit(ref_feat_flat)

    model_filename = save_dir + '/isolation_forest_model.pkl'
    joblib.dump(clf, model_filename)

def agg_prior(prior_image_path, predictor, save_dir, band_choice='optic'):
    # step-1: use gdal to fetch prior image
    prior_image = read_images(prior_image_path, band_choice=band_choices[band_choice], file_tpye='gdal')

    # step-2: use sam predictor to fetch characterization of prior image
    ref_feat = deep_features(prior_image, predictor, layer=-1)
    ref_feat = channel_mean_image(ref_feat, near_radius=2)

    # step-3: agglomerative clustering
    rh, rw, rc = ref_feat.shape
    ref_feat_flat = ref_feat.reshape(rh * rw, rc)

    distance_threshold = 10
    agg_clustering = AgglomerativeClustering(n_clusters=None, metric='manhattan', memory=None, connectivity=None,
                                             compute_full_tree=True,
                                             linkage='complete', distance_threshold=distance_threshold,
                                             compute_distances=False)
    agg_clustering.fit(ref_feat_flat)

    # Step-3: Acquire Needed Stats
    optimal_n_components = agg_clustering.n_clusters_
    lbl_ = agg_clustering.labels_
    mean_lst = []
    std_lst = []

    for i in range(optimal_n_components):
        group_pts = ref_feat_flat[lbl_ == i]
        mean_vec = np.mean(group_pts, axis=0)
        std_val = np.std(group_pts)
        mean_lst.append(mean_vec)
        std_lst.append(std_val)

    # Output Prior
    np.save(save_dir + '/' 'prior_file_ctr.npy', mean_lst)
    np.save(save_dir + '/' + 'prior_file_var.npy', std_lst)

def gmm_prior(prior_image_path, predictor, reduce_dim, save_dir, band_choice='optic', save=True):
    # step-1: use gdal to fetch prior image
    prior_image = read_images(prior_image_path, band_choice=band_choices[band_choice], file_tpye='gdal')

    # step-2: use sam predictor to fetch characterization of prior image
    ref_feat = deep_features(prior_image, predictor, layer=-1)
    ref_feat = channel_mean_image(ref_feat, near_radius=2)

    # step-3: due to gmm is distance-based, it needs dimensional reduction
    rh, rw, rc = ref_feat.shape
    ref_feat_flat = ref_feat.reshape(rh * rw, rc)
    pca = PCA(n_components=reduce_dim)
    ref_feat_dm = pca.fit_transform(ref_feat_flat)

    # step-4: compute optimal n_component for gmm by using bic metrics
    n_components = np.arange(1, 100)

    # Use joblib to parallelize the BIC computation
    bic_values = Parallel(n_jobs=4)(delayed(compute_bic)(n, ref_feat_dm) for n in tqdm(n_components))

    min_bic = min(bic_values)
    optimal_n_components = n_components[bic_values.index(min_bic)]
    print(f"The optimal number of components by BIC is: {optimal_n_components}")

    # Create a Gaussian Mixture Model with the optimal number of components
    optimal_gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=42)
    optimal_gmm.fit(ref_feat_dm)

    lbl_ = optimal_gmm.predict(ref_feat_dm)

    mean_lst = []
    std_lst = []

    for i in range(optimal_n_components):
        group_pts = ref_feat_flat[lbl_ == i]
        mean_vec = np.mean(group_pts, axis=0)
        std_val = np.std(group_pts)
        mean_lst.append(mean_vec)
        std_lst.append(std_val)

    # Output Prior
    if save:
        np.save(save_dir + '/' 'prior_file_ctr.npy', mean_lst)
        np.save(save_dir + '/' + 'prior_file_var.npy', std_lst)
    else:
        return np.array(mean_lst), np.array(std_lst)

def distort_prior(prior_image_path, predictor, reduce_dim, save_dir, band_choice='optic', save=True, d_param = 10):
    # step-1: use gdal to fetch prior image
    ori_prior_image = read_images(prior_image_path, band_choice=band_choices[band_choice], file_tpye='gdal')

    # Add Distortion Here
    prior_image = np.zeros(shape=ori_prior_image.shape, dtype=np.uint8)

    for i in range(prior_image.shape[0]):
        for j in range(prior_image.shape[1]):
            if (i + d_param) < ori_prior_image.shape[0] and (j + d_param) < ori_prior_image.shape[1]:
                prior_image[i, j] = ori_prior_image[i + d_param, j + d_param]

    # step-2: use sam predictor to fetch characterization of prior image
    ref_feat = deep_features(prior_image, predictor, layer=-1)
    ref_feat = channel_mean_image(ref_feat, near_radius=2)

    # step-3: due to gmm is distance-based, it needs dimensional reduction
    rh, rw, rc = ref_feat.shape
    ref_feat_flat = ref_feat.reshape(rh * rw, rc)
    pca = PCA(n_components=reduce_dim)
    ref_feat_dm = pca.fit_transform(ref_feat_flat)

    # step-4: compute optimal n_component for gmm by using bic metrics
    n_components = np.arange(1, 100)

    # Use joblib to parallelize the BIC computation
    bic_values = Parallel(n_jobs=4)(delayed(compute_bic)(n, ref_feat_dm) for n in tqdm(n_components))

    min_bic = min(bic_values)
    optimal_n_components = n_components[bic_values.index(min_bic)]
    print(f"The optimal number of components by BIC is: {optimal_n_components}")

    # Create a Gaussian Mixture Model with the optimal number of components
    optimal_gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=42)
    optimal_gmm.fit(ref_feat_dm)

    lbl_ = optimal_gmm.predict(ref_feat_dm)

    mean_lst = []
    std_lst = []

    for i in range(optimal_n_components):
        group_pts = ref_feat_flat[lbl_ == i]
        mean_vec = np.mean(group_pts, axis=0)
        std_val = np.std(group_pts)
        mean_lst.append(mean_vec)
        std_lst.append(std_val)

    # Output Prior
    if save:
        np.save(save_dir + '/' 'prior_file_ctr.npy', np.array(mean_lst).astype(np.float32))
        np.save(save_dir + '/' + 'prior_file_var.npy', np.array(std_lst).astype(np.float32))
    else:
        return np.array(mean_lst), np.array(std_lst)

if __name__ == '__main__':
    # This is the root dir to save batch priors
    save_root = './fire_batch_agg_prior'
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)

    # Input prior image path here
    batch_prior_path = [
        r'D:\bnu\esad_2023\demo_programs\dataset2\paradise_fire\cliped\paired_win.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\karymsky_volcano\cliped\paired_spr.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\paired_aut.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\mao_landslide\cliped\paired_sum.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_pre.tif',
    ]

    bc = ['fire', 'fire', 'fire', 'fire', 'l7_fire']

    # construct predictor here in case repeat operations
    sam_encoder = construct_feature_extractor()

    # Iteratively generate prior for different events
    for i, prior_path in enumerate(batch_prior_path):
        event_name = prior_path.split('\\')[5]
        save_dir = save_root + '/' + event_name
        os.mkdir(save_dir)
        #gmm_prior(prior_image_path=prior_path, predictor=sam_encoder, reduce_dim=25, save_dir=save_dir, band_choice=bc[i], save=True)
        agg_prior(prior_path, sam_encoder, save_dir, band_choice=bc[i])
        #ifr_prior(prior_path, sam_encoder, save_dir, band_choice=bc[i])