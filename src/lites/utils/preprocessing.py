"""
Utility functions for time series preprocessing and manipulation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal as scipy_signal


def normalize_time_series(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize time series data.

    Args:
        data: Input time series array
        method: Normalization method ('zscore', 'minmax', 'robust')

    Returns:
        Normalized time series array
    """
    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return data - median
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_time_series(
    data: np.ndarray, window_size: int = 5, method: str = "moving_average"
) -> np.ndarray:
    """
    Smooth time series data.

    Args:
        data: Input time series array
        window_size: Size of smoothing window
        method: Smoothing method ('moving_average', 'savgol', 'exponential')

    Returns:
        Smoothed time series array
    """
    if method == "moving_average":
        return np.convolve(data, np.ones(window_size) / window_size, mode="same")
    elif method == "savgol":
        if window_size % 2 == 0:
            window_size += 1
        return scipy_signal.savgol_filter(data, window_size, polyorder=2)
    elif method == "exponential":
        alpha = 2 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def remove_outliers(
    data: np.ndarray, method: str = "zscore", threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from time series data.

    Args:
        data: Input time series array
        method: Outlier detection method ('zscore', 'iqr')
        threshold: Threshold for outlier detection

    Returns:
        Tuple of (cleaned_data, outlier_mask)
    """
    if method == "zscore":
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outlier_mask = z_scores > threshold
    elif method == "iqr":
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        outlier_mask = (data < lower_bound) | (data > upper_bound)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    cleaned_data = data.copy()
    # Replace outliers with interpolated values
    if np.any(outlier_mask):
        valid_indices = np.where(~outlier_mask)[0]
        outlier_indices = np.where(outlier_mask)[0]
        cleaned_data[outlier_indices] = np.interp(
            outlier_indices, valid_indices, data[valid_indices]
        )

    return cleaned_data, outlier_mask


def resample_time_series(
    data: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    target_length: Optional[int] = None,
    target_rate: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample time series data to a different length or rate.

    Args:
        data: Input time series array
        timestamps: Original timestamps (optional)
        target_length: Target length for resampling
        target_rate: Target sampling rate

    Returns:
        Tuple of (resampled_data, resampled_timestamps)
    """
    if timestamps is None:
        timestamps = np.arange(len(data))

    if target_length is not None:
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], target_length)
    elif target_rate is not None:
        duration = timestamps[-1] - timestamps[0]
        target_length = int(duration * target_rate)
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], target_length)
    else:
        raise ValueError("Either target_length or target_rate must be specified")

    resampled_data = np.interp(new_timestamps, timestamps, data)
    return resampled_data, new_timestamps


def calculate_statistics(data: np.ndarray) -> dict:
    """
    Calculate basic statistics for time series data.

    Args:
        data: Input time series array

    Returns:
        Dictionary of statistics
    """
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "length": len(data),
    }
