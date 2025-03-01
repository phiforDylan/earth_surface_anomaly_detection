import os
import shutil
from toolkits import deep_features, construct_feature_extractor, read_images, read_nan_mask, threshold_f1
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
from batch_prior import gmm_prior
from tqdm import tqdm
from feature_filters import channel_mean_image
import numpy as np
from skimage.transform import resize
import matplotlib.cm as cm
import cv2

# parameter control
load_prior = True  # if this is set true, then the prior will be loaded but not computed
band_choices = {
    'optic': [3, 2, 1], 'fire': [5, 4, 2], 'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0], 'l7_veg': [3, 2, 1], 'l7_fire': [4, 3, 2],
    'band3': [2, 1, 0]
}
bc_prior = {'aoraki_landslide': 'band3', 'co_mudslide': 'l7_fire'}
bc_predict = {'aoraki_landslide': 'band3', }
glb_bc = 'fire'
sam_encoder = construct_feature_extractor()

# step-1: load dataset
dataset_root = './tgrs_dataset'
events = os.listdir(dataset_root)
events = ['paradise_fire', 'karymsky_volcano', 'mao_landslide', 'florence_flood', 'co_mudslide']
prior_paths = [dataset_root + '/' + event + '/pre.tif' for event in events]
detection_paths = [dataset_root + '/' + event + '/post.tif' for event in events]

# step-2: construct prior
prior_saver = './prior_base/tgrs_dataset_gmm_prior'
if load_prior is False:
    if os.path.exists(prior_saver):
        shutil.rmtree(prior_saver)
    os.mkdir(prior_saver)
    for prior_path in tqdm(prior_paths):
        event_name = prior_path.split('/')[-2]
        save_dir = prior_saver + '/' + event_name
        os.mkdir(save_dir)
        event_bc = bc_prior[event_name] if event_name in bc_prior.keys() else glb_bc
        gmm_prior(prior_image_path=prior_path, predictor=sam_encoder, reduce_dim=25, save_dir=save_dir,
                  band_choice=event_bc, save=True)

# step-3: anomaly detection
for detect_path in tqdm(detection_paths):
    # read detection image
    event_name = detect_path.split('/')[-2]
    det_bc = bc_predict[event_name] if event_name in bc_predict.keys() else glb_bc
    det_img = read_images(detect_path, band_choice=band_choices[det_bc], file_tpye='gdal')
    h, w, _ = det_img.shape
    nan_mask = read_nan_mask(detect_path)

    # compute features
    det_feat = deep_features(det_img, sam_encoder, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)

    # load data
    center_his = np.load(prior_saver + '/' + event_name + '/prior_file_ctr.npy')
    std_his = np.load(prior_saver + '/' + event_name + '/prior_file_var.npy')

    # compute anomaly score
    det_signal = det_feat.reshape(-1, det_feat.shape[2])
    diminish = np.sum(np.abs(det_signal[:, None, :] - center_his), axis=-1)
    std_his_reshape = std_his.reshape(1, -1)
    weighted_diminish = diminish / std_his_reshape
    sort_simi_var = np.partition(weighted_diminish, 1, axis=1)[:, :1]
    anomaly_scores = np.mean(sort_simi_var, axis=1).reshape(det_feat.shape[0], det_feat.shape[1])
    norm_score = (anomaly_scores - np.mean(anomaly_scores)) / np.sqrt(np.var(anomaly_scores))
    our_anomaly = np.nan_to_num(norm_score, nan=0.)
    final_score = (our_anomaly - np.min(our_anomaly)) / (np.max(our_anomaly) - np.min(our_anomaly))
    final_score = resize(final_score, (h, w), order=1, mode='constant', anti_aliasing=False)
    final_score[nan_mask] = 0.

    # performance evaluation
    label_img_path = dataset_root + '/' + event_name + '/label.png'
    label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
    label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)
    predict_bin = final_score.reshape(1024 * 1024)

    # AP
    average_precision = average_precision_score(label_bin, predict_bin)
    print(event_name, average_precision)

    # P-R-Recall and Effective Range
    thresholds, f1_scores = threshold_f1(predict_bin, label_bin, threshold_gap=0.01, n_jobs=8)
    best_idx = np.argmax(f1_scores)
    f1_max = f1_scores[best_idx]
    optimal_thr = thresholds[best_idx]
    opt_seg = predict_bin > optimal_thr

    # output_optimal_seg
    output_seg = opt_seg.reshape(1024, 1024).astype(np.uint8)
    output_seg = cv2.cvtColor(output_seg * 255, cv2.COLOR_GRAY2BGR)
    output_name = './best_f1_mapping/' + event_name + '_our_bestseg.jpg'
    gt_objs, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_seg, gt_objs, -1, (0, 0, 255), 3)
    cv2.imwrite(output_name, output_seg)

    precision_opt = precision_score(label_bin, opt_seg)
    recall_opt = recall_score(label_bin, opt_seg)
    print('{} - AP: {} - Thr: {} - P: {} - R: {} - F1: {}'.format(event_name, round(average_precision, 3),
                                                                  round(optimal_thr, 3), round(precision_opt, 3),
                                                                  round(recall_opt, 3), round(f1_max, 3)))
    condition = f1_scores >= 0.6
    indices = np.where(condition)
    if indices[0].shape[0] > 0:
        print(np.min(indices) * 0.01, np.max(indices) * 0.01)
    else:
        print('No Effective Threshold Range')