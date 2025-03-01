import numpy as np

def var_image(feature_image, near_radius=2):
    row, col = feature_image.shape
    var_mat = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
            row_min = max(i - near_radius, 0)
            col_min = max(j - near_radius, 0)
            row_max = min(i + near_radius + 1, row)
            col_max = min(j + near_radius + 1, col)
            patch = feature_image[row_min: row_max, col_min: col_max]
            var_mat[i,j] = np.var(patch)
    return var_mat

def mean_image(feature_image, near_radius=2):
    row, col = feature_image.shape
    mean_mat = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
            row_min = max(i - near_radius, 0)
            col_min = max(j - near_radius, 0)
            row_max = min(i + near_radius + 1, row)
            col_max = min(j + near_radius + 1, col)
            patch = feature_image[row_min: row_max, col_min: col_max]
            mean_mat[i,j] = np.mean(patch)
    return mean_mat

def cv_image(feature_image, near_radius=2):
    row, col = feature_image.shape
    cv_mat = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
            row_min = max(i - near_radius, 0)
            col_min = max(j - near_radius, 0)
            row_max = min(i + near_radius + 1, row)
            col_max = min(j + near_radius + 1, col)
            patch = feature_image[row_min: row_max, col_min: col_max]
            cv_mat[i,j] = np.mean(patch) / np.var(patch)
    return cv_mat

def channel_mean_image(feature_image, near_radius=2):
    row, col, channel = feature_image.shape
    mean_mat = np.zeros(shape=(row, col, channel))
    for i in range(channel):
        feaure_i = feature_image[:, :, i]
        mean_mat[:, :, i] = mean_image(feaure_i, near_radius)
    return mean_mat