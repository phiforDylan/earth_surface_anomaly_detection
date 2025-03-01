import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.gmm import GMM
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
import cv2
from toolkits import construct_feature_extractor, deep_features, read_images, read_nan_mask, threshold_f1
from feature_filters import channel_mean_image
import os
from sklearn.metrics import average_precision_score, precision_score, recall_score
from tqdm import tqdm
from skimage.transform import resize

band_choices = {
    'optic': [3, 2, 1], 'fire': [5, 4, 2], 'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0], 'l7_veg': [3, 2, 1], 'l7_fire': [4, 3, 2],
    'band3': [2, 1, 0], 'all': slice(None)
}

def cosine_baseline(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # normalize p_img
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    p_deno = np.linalg.norm(p_flat, ord=2, axis=1)
    p_deno = np.expand_dims(p_deno, axis=-1)
    p_l2norm = p_flat / p_deno
    # normalize d_img
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    d_deno = np.linalg.norm(d_flat, ord=2, axis=1)
    d_deno = np.expand_dims(d_deno, axis=-1)
    d_l2norm = d_flat / d_deno
    # compute cosine similarity
    cos_anomaly = 1 - np.sum(np.multiply(p_l2norm, d_l2norm), axis=1).reshape(h, w)
    cos_anomaly = np.nan_to_num(cos_anomaly, nan=0.)
    cos_anomaly = (cos_anomaly - np.min(cos_anomaly)) / (np.max(cos_anomaly) - np.min(cos_anomaly))
    return cos_anomaly

def euclidean_baseline(prior_path, detection_path, pband_choice, dband_choice):
    #print(prior_path)
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])

    # compute metrics
    eu_anomaly = np.sqrt(np.sum(((p_img - d_img)**2), axis=-1))
    eu_anomaly = np.nan_to_num(eu_anomaly, nan=0.)
    eu_anomaly = (eu_anomaly - np.min(eu_anomaly)) / (np.max(eu_anomaly) - np.min(eu_anomaly))
    return eu_anomaly

def knn_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = KNN(method='largest', n_neighbors=5)
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    return score_a

def ifr_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = IForest()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    return score_a

def gmm_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = GMM()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    return score_a

def pca_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = PCA()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    return score_a

def hbos_anomaly(prior_path, detection_path, pband_choice, dband_choice):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape
    # reshape images
    p_flat = np.reshape(p_img, (p_img.shape[0] * p_img.shape[1], p_img.shape[2]))
    d_flat = np.reshape(d_img, (d_img.shape[0] * d_img.shape[1], d_img.shape[2]))
    # prior computation
    clf = HBOS()
    clf.fit(p_flat)
    score = clf.predict_proba(d_flat, method='linear')
    score = np.array(score)
    score_a = np.reshape(score[:, 1], (h, w))
    return score_a

def cosine_embedding(prior_path, detection_path, pband_choice, dband_choice, encoder):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape

    # construct predictor
    predictor = encoder

    # extract features
    det_feat = deep_features(d_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)
    pri_feat = deep_features(p_img, predictor, layer=-1)
    pri_feat = channel_mean_image(pri_feat, near_radius=2)
    fh, fw, _ = det_feat.shape

    # cosine baseline
    p_flat = np.reshape(pri_feat, (pri_feat.shape[0] * pri_feat.shape[1], pri_feat.shape[2]))
    p_deno = np.linalg.norm(p_flat, ord=2, axis=1)
    p_deno = np.expand_dims(p_deno, axis=-1)
    p_l2norm = p_flat / p_deno

    d_flat = np.reshape(det_feat, (det_feat.shape[0] * det_feat.shape[1], det_feat.shape[2]))
    d_deno = np.linalg.norm(d_flat, ord=2, axis=1)
    d_deno = np.expand_dims(d_deno, axis=-1)
    d_l2norm = d_flat / d_deno

    cos_anomaly = 1 - np.sum(np.multiply(p_l2norm, d_l2norm), axis=1).reshape(fh, fw)
    cos_anomaly = np.nan_to_num(cos_anomaly, nan=0.)
    cos_anomaly = (cos_anomaly - np.min(cos_anomaly)) / (np.max(cos_anomaly) - np.min(cos_anomaly))
    return cos_anomaly

def euclidean_embedding(prior_path, detection_path, pband_choice, dband_choice, encoder):
    p_img = read_images(prior_path, band_choice=band_choices[pband_choice])
    d_img = read_images(detection_path, band_choice=band_choices[dband_choice])
    h, w, _ = d_img.shape

    # construct predictor
    predictor = encoder

    # extract features
    det_feat = deep_features(d_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)
    pri_feat = deep_features(p_img, predictor, layer=-1)
    pri_feat = channel_mean_image(pri_feat, near_radius=2)
    fh, fw, _ = det_feat.shape

    # compute metrics
    eu_anomaly = np.sqrt(np.sum(((det_feat - pri_feat) ** 2), axis=-1))
    eu_anomaly = np.nan_to_num(eu_anomaly, nan=0.)
    eu_anomaly = (eu_anomaly - np.min(eu_anomaly)) / (np.max(eu_anomaly) - np.min(eu_anomaly))
    return eu_anomaly

if __name__ == '__main__':
    compare_fn = [cosine_baseline, euclidean_baseline, gmm_anomaly, pca_anomaly, hbos_anomaly, ifr_anomaly]
    compare_fn2 = [cosine_embedding, euclidean_embedding]

    bc_prior = {'aoraki_landslide':'band3', 'co_mudslide': 'l7_fire'}
    bc_predict = {'aoraki_landslide':'band3', 'co_mudslide': 'fire'}
    glb_bc = 'all'
    glb_bc_dcd = 'fire'

    sam_encoder = construct_feature_extractor()

    dataset_root = './tgrs_dataset'
    #events = os.listdir(dataset_root)
    events = ['paradise_fire', 'karymsky_volcano', 'mao_landslide', 'florence_flood', 'co_mudslide']
    prior_paths = [dataset_root + '/' + event + '/pre.tif' for event in events]
    detection_paths = [dataset_root + '/' + event + '/post.tif' for event in events]
    total_ap = {}

    for i, event in tqdm(enumerate(events)):
        event_ap = []
        prior_bc = bc_prior[event] if event in bc_prior.keys() else glb_bc
        prior_bc_dcd = bc_prior[event] if event in bc_prior.keys() else glb_bc_dcd
        det_bc = bc_predict[event] if event in bc_predict.keys() else glb_bc
        det_bc_dcd = bc_predict[event] if event in bc_predict.keys() else glb_bc_dcd
        prior_path = prior_paths[i]
        detect_path = detection_paths[i]
        nan_mask = read_nan_mask(detect_path)
        for fn in compare_fn:
            score = fn(prior_path, detect_path, prior_bc, det_bc)
            score[nan_mask] = 0.
            label_img_path = dataset_root + '/' + event + '/label.png'
            label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
            label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)
            predict_bin = score.reshape(1024 * 1024)
            # AP
            average_precision = average_precision_score(label_bin, predict_bin)
            # Other
            thresholds, f1_scores = threshold_f1(predict_bin, label_bin, threshold_gap=0.01, n_jobs=8)
            best_idx = np.argmax(f1_scores)
            f1_max = f1_scores[best_idx]
            optimal_thr = thresholds[best_idx]
            opt_seg = predict_bin > optimal_thr

            # output_optimal_seg
            output_seg = opt_seg.reshape(1024, 1024).astype(np.uint8)
            output_seg = cv2.cvtColor(output_seg * 255, cv2.COLOR_GRAY2BGR)
            output_name = './best_f1_mapping/' + event + '_' + fn.__name__ + '_bestseg.jpg'
            gt_objs, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_seg, gt_objs, -1, (0, 0, 255), 3)
            cv2.imwrite(output_name, output_seg)

            precision_opt = precision_score(label_bin, opt_seg)
            recall_opt = recall_score(label_bin, opt_seg)
            f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)
            print(fn.__name__)
            print('{} - AP: {} - P: {} - R: {} - F1: {}'.format(event, round(average_precision, 3),
                                                                          round(precision_opt, 3),
                                                                          round(recall_opt, 3), round(f1_opt, 3)))
            condition = f1_scores >= 0.6
            indices = np.where(condition)
            if indices[0].shape[0] > 0:
                print(np.min(indices) * 0.01, np.max(indices) * 0.01)
            else:
                print('No Effective Threshold Range')

        for fn2 in compare_fn2:
            score = fn2(prior_path, detect_path, prior_bc_dcd, det_bc_dcd, sam_encoder)
            # resize score map to 1024
            score = resize(score, (1024, 1024), order=1, mode='constant', anti_aliasing=False)
            score[nan_mask] = 0.

            # evaluation
            label_img_path = dataset_root + '/' + event + '/label.png'
            label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
            label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)
            predict_bin = score.reshape(1024 * 1024)
            # AP
            average_precision = average_precision_score(label_bin, predict_bin)
            # Other
            thresholds, f1_scores = threshold_f1(predict_bin, label_bin, threshold_gap=0.01, n_jobs=8)
            best_idx = np.argmax(f1_scores)
            f1_max = f1_scores[best_idx]
            optimal_thr = thresholds[best_idx]
            opt_seg = predict_bin > optimal_thr

            # output_optimal_seg
            output_seg = opt_seg.reshape(1024, 1024).astype(np.uint8)
            output_seg = cv2.cvtColor(output_seg * 255, cv2.COLOR_GRAY2BGR)
            output_name = './best_f1_mapping/' + event + '_' + fn2.__name__ + '_bestseg.jpg'
            gt_objs, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output_seg, gt_objs, -1, (0, 0, 255), 3)
            cv2.imwrite(output_name, output_seg)

            precision_opt = precision_score(label_bin, opt_seg)
            recall_opt = recall_score(label_bin, opt_seg)
            f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)
            print(fn2.__name__)
            print('{} - AP: {} - P: {} - R: {} - F1: {}'.format(event, round(average_precision, 3),
                                                                round(precision_opt, 3),
                                                                round(recall_opt, 3), round(f1_opt, 3)))
            condition = f1_scores >= 0.6
            indices = np.where(condition)
            if indices[0].shape[0] > 0:
                print(np.min(indices) * 0.01, np.max(indices) * 0.01)
            else:
                print('No Effective Threshold Range')









