import matplotlib.pyplot as plt
from skimage.transform import resize
from batch_prior import gmm_prior, agg_prior
from toolkits import read_images, construct_feature_extractor, deep_features, read_nan_mask
from sklearn.metrics import precision_recall_curve, average_precision_score
from feature_filters import channel_mean_image
import numpy as np
import cv2

band_choices = {
    'optic': [3, 2, 1],
    'fire': [5, 4, 2],
    'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0],
    'l7_veg': [3, 2, 1],
    'l7_fire': [4, 3, 2],
}

# dataset for papers

detection_paths = [
        r'D:\bnu\esad_2023\demo_programs\dataset2\paradise_fire\cliped\ca_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\karymsky_volcano\cliped\karymsky_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\flr_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\mao_landslide\cliped\mao_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_post.tif',]

agg_priors = [
    r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\fire_batch_agg_prior\paradise_fire',
    r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\fire_batch_agg_prior\karymsky_volcano',
    r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\fire_batch_agg_prior\florence_flood',
    r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\fire_batch_agg_prior\mao_landslide',
    r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\fire_batch_agg_prior\co_mudslide',
]

label_paths = [
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\paradise_fire.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\karymsky_volcano.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\flr_flood.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\mao_landslide.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\co_mudslide.png',
    ]

for i,detect_path in enumerate(detection_paths):
    b_choice = 'fire'

    sam_encoder = construct_feature_extractor()
    # Feature Compute for Detection Image
    det_img = read_images(detect_path, band_choice=band_choices[b_choice], file_tpye='gdal')
    h, w, _ = det_img.shape
    nan_mask = read_nan_mask(detect_path)

    plt.subplot(121)
    plt.imshow(det_img)

    det_feat = deep_features(det_img, sam_encoder, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)

    print('finish feature compute')

    center_his = np.load(agg_priors[i] + '/prior_file_ctr.npy')
    std_his = np.load(agg_priors[i] + '/prior_file_var.npy')

    # Compute Anomaly Score Here
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
    predict_bin = final_score.reshape(1024 * 1024)

    label_img = cv2.imread(label_paths[i], cv2.IMREAD_GRAYSCALE)
    label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)
    average_precision = average_precision_score(label_bin, predict_bin)
    print(average_precision)

    plt.subplot(122)
    plt.imshow(final_score, cmap='jet')
    plt.show()



