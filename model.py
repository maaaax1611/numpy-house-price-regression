"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    # create float copy of X
    x = np.asarray(X, dtype=float, copy=True)
    # calc mean for every column ignoring NaN --> (,F) 
    means = np.nanmean(x, axis=0)
    # create a mask where the NaNs are in our "matrix" 
    nan_mask = np.isnan(x)
    # replace each NaN with its column/feature mean value
    x = np.where(nan_mask, means, x)
    # now we need to set columns that are all NaN to zero
    nan__col_mask = np.isnan(means)
    # "where the entire column in nan we write 0.0, else we leave x"
    x = np.where(nan__col_mask, 0.0, x)
    return x

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    x = np.asarray(X, dtype=float, copy=True)
    # compute percentiles
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    # compute inter quartile range
    IQR = q3 - q1
    # compute upper and lower bounds
    lower = q1 - k*IQR
    upper = q3 + k*IQR
    return (lower, upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    if numerator.shape != denominator.shape:
        return ValueError
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    # (n,) --> (n,1)
    col_2d = col.reshape(-1, 1)
    return np.hstack((X, col_2d))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    # extract unique labels
    uniques = np.unique(labels)
    # build binary mask
    # trick: broacasting
    # lables (N,1) -- uniques(1,K) --> (N,K)
    mask = labels[:, None] == uniques[None, :]
    # build one hot encoded features
    one_hot = np.where(mask, 1.0, 0.0).astype(float)
    return one_hot

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    means = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # sanitize standard deviations -- we dont want zero variances
    std = np.where(std == 0, 1.0, std)
    return (means, std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    rows, cols = X.shape
    ones = np.ones((rows, 1))
    X = np.hstack((ones, X))
    return X

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    randomizer = np.random.RandomState(seed)
    shuffled = randomizer.permutation(n_samples)
    return shuffled

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    # A * theta = b
    A = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(A, b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    # check for shapes
    if X.shape[1] != theta.shape[0]:
        raise ValueError

    y_hat = X @ theta
    return y_hat

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true - y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    residuals = (y_true - y_pred) ** 2
    mean = np.mean(residuals)
    return np.sqrt(mean)

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # calc
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    r_sqrd = 1 - (ss_res / ss_tot)
    return r_sqrd

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    residuals = y_true - y_pred
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    res_median_abs = np.median(np.abs(residuals))
    return {'mean': res_mean, 'std': res_std, 'median_abs': res_median_abs}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X, iqr_k)
    X_clipped = clip_columns(X, lower, upper)
    return X_clipped

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    num = X_num[:, ratio_num_idx]
    den = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(num, den)
    X = append_column(X_num, ratio)
    if cat_labels is not None:
        cat_one_hot = one_hot_encode(cat_labels)
        X = np.hstack((X, cat_one_hot))
    return X

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_samples = X.shape[0]
    shuffle_idx = make_shuffled_indices(n_samples, seed)
    train_idx, val_idx, test_idx = partition_indices(shuffle_idx, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    return {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

