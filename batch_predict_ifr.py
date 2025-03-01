import os.path
import shutil
import joblib
from toolkits import construct_feature_extractor, deep_features, read_images, read_nan_mask
from feature_filters import channel_mean_image, mean_image
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize

band_choices = {
    'optic': [3, 2, 1], 'fire': [5, 4, 2], 'veg': [4, 3, 2],
    'l7_optic': [2, 1, 0], 'l7_veg': [3, 2, 1], 'l7_fire': [4, 3, 2],
    'band3': [2, 1, 0]
}

def ifr_result_generation(p_path, det_path, label_path, band_choice='optic'):
    plt.figure(figsize=(10, 6))

    # Step-1 Read Detection Image
    det_img = read_images(det_path, band_choice=band_choices[band_choice], file_tpye='gdal')
    h, w, c = det_img.shape
    nan_mask = read_nan_mask(det_path)

    # Step-2 Feature Extraction
    predictor = construct_feature_extractor()
    det_feat = deep_features(det_img, predictor, layer=-1)
    det_feat = channel_mean_image(det_feat, near_radius=2)
    det_feat_flat = det_feat.reshape(64 * 64, 256)

    # Step-3 Load Priors
    clf_loaded = joblib.load(p_path)

    # Step-4 Generate Maps
    score = clf_loaded.predict_proba(det_feat_flat, method='linear')
    score_a = np.reshape(score[:, 1], (64, 64))
    if score_a.shape[0] != 1024:
        norm_score_map = resize(score_a, (1024, 1024), order=1, mode='constant', anti_aliasing=False)
    else:
        norm_score_map = score_a
    norm_score_map[nan_mask] = 0.
    predict_bin = norm_score_map.reshape(1024 * 1024)
    plt.subplot(121)
    plt.imshow(norm_score_map, cmap='jet')


    # Step-5 Compute AP
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(122)
    plt.imshow(label_img)
    plt.show()
    label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)
    average_precision = average_precision_score(label_bin, predict_bin)
    print(average_precision)

if __name__ == '__main__':
    score_path = './score'
    if os.path.exists(score_path):
        shutil.rmtree(score_path)
    os.mkdir(score_path)

    # dataset for papers
    prior_dirs = [
        './fire_batch_ifr_prior/paradise_fire/isolation_forest_model.pkl',
        './fire_batch_ifr_prior/karymsky_volcano/isolation_forest_model.pkl',
        './fire_batch_ifr_prior/florence_flood/isolation_forest_model.pkl',
        './fire_batch_ifr_prior/mao_landslide/isolation_forest_model.pkl',
        './fire_batch_ifr_prior/co_mudslide/isolation_forest_model.pkl',]

    detection_paths = [
        r'D:\bnu\esad_2023\demo_programs\dataset2\paradise_fire\cliped\ca_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\karymsky_volcano\cliped\karymsky_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\flr_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\mao_landslide\cliped\mao_post.tif',
        r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_post.tif',]

    label_paths = [
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\paradise_fire.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\karymsky_volcano.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\flr_flood.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\mao_landslide.png',
        r'D:\bnu\esad_2023\demo_programs\ESAD_TGRS\gt_polygon\co_mudslide.png',
    ]

    # generate a predictor here
    predictor = construct_feature_extractor()

    bc = ['fire', 'fire', 'fire', 'fire', 'fire']

    # automatic run
    for i in range(len(detection_paths)):
        prior_dir = prior_dirs[i]
        detection_path = detection_paths[i]
        label_path = label_paths[i]
        ifr_result_generation(prior_dir, detection_path, label_path, bc[i])
