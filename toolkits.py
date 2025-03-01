import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from SAM import SamPredictor, sam_model_registry
from osgeo import gdal
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

def gdalarr_to_img(img, band_choice=[0, 1, 2]):
    img = np.nan_to_num(img, nan=0.)
    band_max = np.max(img, axis=(1, 2))
    band_min = np.min(img, axis=(1, 2))
    band_max = np.expand_dims(np.expand_dims(band_max, axis=-1), axis=-1)
    band_min = np.expand_dims(np.expand_dims(band_min, axis=-1), axis=-1)
    img_norm = (img - band_min) / (band_max - band_min)
    img_norm = img_norm * 255.
    img_norm = img_norm.astype(np.uint8)
    img_optic = img_norm[band_choice, ...].transpose(1, 2, 0)
    return img_optic

def read_images(image_path, band_choice, file_tpye='gdal'):
    assert file_tpye in ['gdal', 'npy']
    if file_tpye == 'gdal':
        img = gdal.Open(image_path).ReadAsArray()
    elif file_tpye == 'npy':
        img = np.load(image_path)
    img = np.nan_to_num(img, nan=0.)
    band_max = np.max(img, axis=(1, 2))
    band_min = np.min(img, axis=(1, 2))
    band_max = np.expand_dims(np.expand_dims(band_max, axis=-1), axis=-1)
    band_min = np.expand_dims(np.expand_dims(band_min, axis=-1), axis=-1)
    img_norm = (img - band_min) / (band_max - band_min)
    img_norm = img_norm * 255.
    img_norm = img_norm.astype(np.uint8)
    if band_choice != None:
        result_img = img_norm[band_choice, ...].transpose(1, 2, 0)
    else:
        result_img = img_norm.transpose(1, 2, 0)
    return result_img

def read_nan_mask(image_path, file_tpye='gdal'):
    assert file_tpye in ['gdal', 'npy']
    if file_tpye == 'gdal':
        img = gdal.Open(image_path).ReadAsArray()
    elif file_tpye == 'npy':
        img = np.load(image_path)
    mask = ~np.all(img, axis=0)
    return mask

def deep_features(image, sam_predictor, layer=-1):
    sam_predictor.set_image(image)
    deep_feat = sam_predictor.features[layer]
    return np.array(deep_feat.squeeze(0).permute(1, 2, 0).detach().cpu())

def compute_euclidean(img1, img2):
    return np.sqrt(np.sum((img1 - img2) ** 2, axis=2))

def apply_tsne(array, reduced_dim=1):
    standardized_data = StandardScaler().fit_transform(array)
    tsne = TSNE(n_components=reduced_dim)
    transformed_data = tsne.fit_transform(standardized_data)
    return transformed_data

def construct_feature_extractor(path = r'D:\bnu\esad_2023\demo_programs\ESAD_V2312\checkpoints\sam_vit_h_4b8939.pth'):
    sam_checkpoint = path
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device='cuda')
    predictor = SamPredictor(sam)
    return predictor

def get_coord(path):
    dataset = gdal.Open(path)
    return dataset.GetGeoTransform()

def gdal_open(path):
    return gdal.Open(path)

def pixel_to_geographic(x, y, geotransform):
    x_geo = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    y_geo = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return x_geo, y_geo

def compute_f1_for_threshold(threshold, score_map, label_image):
    # Binarize the score map using the given threshold
    binary_prediction = (score_map >= threshold).astype(int)
    # Compute the F1 score
    f1 = f1_score(label_image.flatten(), binary_prediction.flatten())
    return threshold, f1
def threshold_f1(score_map, label_image, threshold_gap=0.01, n_jobs=6):
    thresholds = np.arange(0, 1, threshold_gap)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_f1_for_threshold)(threshold, score_map, label_image) for threshold in thresholds
    )
    results_np = np.array(results)
    threshold = results_np[:, 0]
    f1_scores = results_np[:, 1]
    return threshold, f1_scores

if __name__ == '__main__':
    # Sample score_map and label_image for demonstration
    score_map = np.random.rand(100, 100)  # Example probability scores
    label_image = (np.random.rand(100, 100) > 0.5).astype(int)  # Example ground truth labels

    threshold, f1_scores = threshold_f1(score_map, label_image, threshold_gap=0.01, n_jobs=-1)

