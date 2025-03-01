import os.path
import shutil

from toolkits import construct_feature_extractor, deep_features, read_images
from feature_filters import channel_mean_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

band_choices = {
    'optic': [3, 2, 1],
    'fire': [5, 4, 2],
    'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0],
    'l7_veg': [3, 2, 1],
    'l7_fire': [4, 3, 2],
    'l8': [5, 4, 3, 2, 1],
    'l7': [4, 3, 2, 1, 0],
}

def only_rep_predict(p_dir, det_path, band_choice='optic', save_name=None):
    plt.figure(figsize=(10, 6))

    # Step-1 Read Detection Image
    det_img = read_images(det_path, band_choice=band_choices[band_choice], file_tpye='gdal')
    h, w, c = det_img.shape

    # Step-2 Feature Extraction
    predictor = construct_feature_extractor()
    det_feat = deep_features(det_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)

    # Step-3 Load Priors
    center_his = np.load(p_dir + '/prior_file_ctr.npy')

    # Step-4 Detection
    det_signal = det_feat.reshape(-1, det_feat.shape[2])
    diminish = np.sum(np.abs(det_signal[:, None, :] - center_his), axis=-1)
    sort_simi_var = np.partition(diminish, 1, axis=1)[:, :1]
    anomaly_scores = np.mean(sort_simi_var, axis=1).reshape(det_feat.shape[0], det_feat.shape[1])
    norm_score = (anomaly_scores - np.mean(anomaly_scores)) / np.sqrt(np.var(anomaly_scores))
    our_anomaly = np.nan_to_num(norm_score, nan=0.)
    final_score = (our_anomaly - np.min(our_anomaly)) / (np.max(our_anomaly) - np.min(our_anomaly))
    np.save(save_name, final_score)
    """plt.imshow(final_score, cmap='jet')
    plt.show()"""

    # Step-5 Storage final score
    #case_name = p_dir.split('/')[-1]
    #np.save('./score_maps/' + case_name + '/only_rep_result.npy', final_score)

def p_weighted_predict(p_dir, det_path, band_choice='optic'):
    plt.figure(figsize=(10, 6))

    # Step-1 Read Detection Image
    det_img = read_images(det_path, band_choice=band_choices[band_choice], file_tpye='gdal')
    h, w, c = det_img.shape

    # Step-2 Feature Extraction
    predictor = construct_feature_extractor()
    det_feat = deep_features(det_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)

    # Step-3 Load Priors
    center_his = np.load(p_dir + '/prior_file_ctr.npy')
    std_his = np.load(p_dir + '/prior_file_var.npy')
    p_his = np.load(p_dir + '/prior_file_pweight.npy')

    # Step-4 Detection
    det_signal = det_feat.reshape(-1, det_feat.shape[2])
    diminish = np.sum(np.abs(det_signal[:, None, :] - center_his), axis=-1)

    std_his_reshape = std_his.reshape(1, -1)
    p_his_reshape = p_his.reshape(1, -1)

    weighted_diminish = diminish / (std_his_reshape * p_his_reshape)

    sort_simi_var = np.partition(weighted_diminish, 1, axis=1)[:, :1]
    anomaly_scores = np.mean(sort_simi_var, axis=1).reshape(det_feat.shape[0], det_feat.shape[1])
    norm_score = (anomaly_scores - np.mean(anomaly_scores)) / np.sqrt(np.var(anomaly_scores))
    our_anomaly = np.nan_to_num(norm_score, nan=0.)
    final_score = (our_anomaly - np.min(our_anomaly)) / (np.max(our_anomaly) - np.min(our_anomaly))

    case_name = p_dir.split('/')[-1]
    os.mkdir('./score/' + case_name)
    np.save('./score/' + case_name + '/p_weighted_result.npy', final_score)

def std_predict(p_dir, det_path, band_choice='optic', save_name=None):

    # Step-1 Read Detection Image
    det_img = read_images(det_path, band_choice=band_choices[band_choice], file_tpye='gdal')
    h, w, c = det_img.shape

    # Step-2 Feature Extraction
    predictor = construct_feature_extractor()
    det_feat = deep_features(det_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)

    # Step-3 Load Priors
    center_his = np.load(p_dir + '/prior_file_ctr.npy')
    std_his = np.load(p_dir + '/prior_file_var.npy')

    # Step-4 Detection
    det_signal = det_feat.reshape(-1, det_feat.shape[2])
    diminish = np.sum(np.abs(det_signal[:, None, :] - center_his), axis=-1)

    std_his_reshape = std_his.reshape(1, -1)
    weighted_diminish = diminish / std_his_reshape

    sort_simi_var = np.partition(weighted_diminish, 1, axis=1)[:, :1]
    anomaly_scores = np.mean(sort_simi_var, axis=1).reshape(det_feat.shape[0], det_feat.shape[1])
    norm_score = (anomaly_scores - np.mean(anomaly_scores)) / np.sqrt(np.var(anomaly_scores))
    our_anomaly = np.nan_to_num(norm_score, nan=0.)
    final_score = (our_anomaly - np.min(our_anomaly)) / (np.max(our_anomaly) - np.min(our_anomaly))
    np.save(save_name, final_score)


if __name__ == '__main__':
    score_path = './ifr_results'
    if os.path.exists(score_path):
        shutil.rmtree(score_path)
    os.mkdir(score_path)
    events = ['paradise_fire', 'karymsky_volcano', 'florence_flood', 'mao_landslide', 'co_mudslide']
    modalities = ['optic', 'veg', 'fire']

    for prior_modal in modalities:
        prior_dirs = ['./' + prior_modal + '_batch_gmm_prior/' + event  for event in events]
        detection_paths = [
            r'D:\bnu\esad_2023\demo_programs\dataset2\paradise_fire\cliped\ca_post.tif',
            r'D:\bnu\esad_2023\demo_programs\dataset2\karymsky_volcano\cliped\karymsky_post.tif',
            r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\flr_post.tif',
            r'D:\bnu\esad_2023\demo_programs\dataset2\mao_landslide\cliped\mao_post.tif',
            r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_post.tif', ]
        for detection_modal in modalities:
            save_root = score_path + '/' + prior_modal + '_' + detection_modal
            os.mkdir(save_root)
            for i in range(len(detection_paths)):
                save_name = save_root + '/' + events[i] + '.npy'
                prior_dir = prior_dirs[i]
                detection_path = detection_paths[i]
                std_predict(prior_dir, detection_path, detection_modal, save_name)