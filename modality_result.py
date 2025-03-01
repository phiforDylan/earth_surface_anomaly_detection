# Given the results of a method, plot its evaluation metric on different cases
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from toolkits import read_nan_mask
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from shapely.geometry import Polygon


def pixel_level_assess(label_img, score_map, nan_mask):
    # binary labels
    label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)

    # resized score maps
    if score_map.shape[0] != 1024:
        norm_score_map = resize(score_map, (1024, 1024), order=1, mode='constant', anti_aliasing=False)
    else:
        norm_score_map = score_map
    # this is to remove nan to 0. problem, especially for methods involve angle computation
    norm_score_map[nan_mask] = 0.

    predict_bin = norm_score_map.reshape(1024 * 1024)
    ap = average_precision_score(label_bin, predict_bin)
    return ap

def extract_best_threshold(label_img, score_map, nan_mask, threshold_gap=0.05):
    thresholds = np.arange(0, 1, threshold_gap)

    # binary labels
    label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8).reshape(1024 * 1024)

    # resized score maps
    if score_map.shape[0] != 1024:
        norm_score_map = resize(score_map, (1024, 1024), order=1, mode='constant', anti_aliasing=False)
    else:
        norm_score_map = score_map
    # this is to remove nan to 0. problem, especially for methods involve angle computation
    norm_score_map[nan_mask] = 0.
    predict_bin = norm_score_map.reshape(1024 * 1024)

    best_f1 = 0.
    best_threshold = 0.
    for threshold in thresholds:
        thr_result = predict_bin > threshold
        f1 = f1_score(label_bin, thr_result)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def extract_by_thr(label_img, score_map, nan_mask, thr):
    # This is to extract gts boundaries.
    label_bin = np.where(label_img > 0, 1, 0).astype(np.uint8)
    gt_objs, _ = cv2.findContours(label_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # This is to extract predicted boundaries.
    if score_map.shape[0] != 1024:
        norm_score_map = resize(score_map, (1024, 1024), order=1, mode='constant', anti_aliasing=False)
    else:
        norm_score_map = score_map

    norm_score_map[nan_mask] = 0.

    predict_bin = norm_score_map
    bin_seg = predict_bin > thr
    plt.imshow(bin_seg, cmap='gray')
    plt.show()

if __name__ == '__main__':
    events = ['paradise_fire', 'karymsky_volcano', 'florence_flood', 'mao_landslide', 'co_mudslide']

    label_paths = {
        'paradise_fire': './gt_polygon/paradise_fire.png',
        'karymsky_volcano': './gt_polygon/karymsky_volcano.png',
        'florence_flood': './gt_polygon/flr_flood.png',
        'mao_landslide': './gt_polygon/mao_landslide.png',
        'co_mudslide': './gt_polygon/co_mudslide.png',
    }

    nan_mask_sources = {
        'paradise_fire': r'D:\bnu\esad_2023\demo_programs\dataset2\paradise_fire\cliped\paired_win.tif',
        'karymsky_volcano': r'D:\bnu\esad_2023\demo_programs\dataset2\karymsky_volcano\cliped\paired_spr.tif',
        'florence_flood': r'D:\bnu\esad_2023\demo_programs\dataset2\florence_flood\cliped\paired_aut.tif',
        'mao_landslide': r'D:\bnu\esad_2023\demo_programs\dataset2\mao_landslide\cliped\paired_sum.tif',
        'co_mudslide': r'D:\bnu\esad_2023\demo_programs\dataset2\co_mudslide\cliped\co_pre.tif'}

    modalities = ['fire',]

    for event in events:
        label_path = label_paths[event]
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        nan_mask_path = nan_mask_sources[event]
        nan_mask = read_nan_mask(nan_mask_path)

        for prior_modal in modalities:
            for detection_modal in modalities:
                score_map_path = './modality_scoremap/' + prior_modal + '_' + detection_modal + '/' + event + '.npy'
                score_map = np.load(score_map_path)
                ap = round(pixel_level_assess(label_img, score_map, nan_mask), 3)
                print(event, prior_modal, detection_modal, ap)
