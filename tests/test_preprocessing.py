"""Tests for preprocessing utilities."""

import numpy as np
import pytest
from lites.utils import preprocessing


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_normalize_zscore(self):
        """Test z-score normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized = preprocessing.normalize_time_series(data, method="zscore")

        assert normalized.shape == data.shape
        assert abs(np.mean(normalized)) < 1e-10  # Mean should be ~0
        assert abs(np.std(normalized) - 1.0) < 1e-10  # Std should be ~1

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized = preprocessing.normalize_time_series(data, method="minmax")

        assert normalized.shape == data.shape
        assert abs(np.min(normalized)) < 1e-10  # Min should be 0
        assert abs(np.max(normalized) - 1.0) < 1e-10  # Max should be 1

    def test_normalize_robust(self):
        """Test robust normalization."""
        data = np.array([1, 2, 3, 4, 5, 100])  # With outlier
        normalized = preprocessing.normalize_time_series(data, method="robust")

        assert normalized.shape == data.shape

    def test_smooth_moving_average(self):
        """Test moving average smoothing."""
        data = np.array([1, 5, 1, 5, 1, 5, 1])
        smoothed = preprocessing.smooth_time_series(data, window_size=3, method="moving_average")

        assert smoothed.shape == data.shape
        # Smoothed version should be less variable
        assert np.std(smoothed) < np.std(data)

    def test_smooth_savgol(self):
        """Test Savitzky-Golay smoothing."""
        data = np.array([1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1])
        smoothed = preprocessing.smooth_time_series(data, window_size=5, method="savgol")

        assert smoothed.shape == data.shape

    def test_smooth_exponential(self):
        """Test exponential smoothing."""
        data = np.array([1, 5, 1, 5, 1, 5, 1])
        smoothed = preprocessing.smooth_time_series(data, window_size=3, method="exponential")

        assert smoothed.shape == data.shape

    def test_remove_outliers_zscore(self):
        """Test outlier removal with z-score."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
        cleaned, mask = preprocessing.remove_outliers(data, method="zscore", threshold=2.0)

        assert cleaned.shape == data.shape
        assert mask[5]  # Last element should be marked as outlier
        assert cleaned[5] != 100  # Outlier should be replaced

    def test_remove_outliers_iqr(self):
        """Test outlier removal with IQR."""
        data = np.array([1, 2, 3, 4, 5, 100])
        cleaned, mask = preprocessing.remove_outliers(data, method="iqr", threshold=1.5)

        assert cleaned.shape == data.shape
        assert np.any(mask)  # Should detect outliers

    def test_resample_by_length(self):
        """Test resampling to specific length."""
        data = np.array([1, 2, 3, 4, 5])
        resampled, new_timestamps = preprocessing.resample_time_series(
            data, target_length=10
        )

        assert len(resampled) == 10
        assert len(new_timestamps) == 10

    def test_resample_by_rate(self):
        """Test resampling to specific rate."""
        data = np.array([1, 2, 3, 4, 5])
        timestamps = np.array([0, 1, 2, 3, 4])
        resampled, new_timestamps = preprocessing.resample_time_series(
            data, timestamps=timestamps, target_rate=2.0
        )

        assert len(resampled) > 0
        assert len(new_timestamps) > 0

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])
        stats = preprocessing.calculate_statistics(data)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["length"] == 5

    def test_invalid_normalization_method(self):
        """Test that invalid normalization method raises error."""
        with pytest.raises(ValueError):
            preprocessing.normalize_time_series(np.array([1, 2, 3]), method="invalid")

    def test_invalid_smoothing_method(self):
        """Test that invalid smoothing method raises error."""
        with pytest.raises(ValueError):
            preprocessing.smooth_time_series(np.array([1, 2, 3]), method="invalid")
