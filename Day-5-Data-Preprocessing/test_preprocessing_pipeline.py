"""
Unit tests for Day 5 preprocessing pipeline. 

Run:
    pytest Day-5-Data-Preprocessing/tests/test_preprocessing_pipeline.py

Or with unittest:
    python -m unittest Day-5-Data-Preprocessing/tests/test_preprocessing_pipeline. py
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent. parent))

from preprocessing_pipeline import (
    create_toy_dataset,
    build_preprocessing_pipeline,
)

import numpy as np
import pandas as pd


class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.df = create_toy_dataset(n_samples=100, random_state=42)

    def test_dataset_creation(self):
        """Test that toy dataset is created with expected shape and columns."""
        self.assertEqual(len(self.df), 100)
        self.assertIn("target", self.df.columns)
        self.assertIn("num_0", self.df.columns)
        self.assertIn("cat_small", self.df.columns)
        self.assertIn("cat_med", self.df.columns)

    def test_missing_values_present(self):
        """Test that missing values are injected as expected."""
        self.assertTrue(self.df.isnull().sum().sum() > 0, "Dataset should have missing values")

    def test_preprocessing_pipeline_transform(self):
        """Test that preprocessing pipeline transforms data without errors."""
        numeric_features = [c for c in self.df.columns if c. startswith("num_")]
        categorical_features = ["cat_small", "cat_med"]

        preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

        X = self.df. drop(columns=["target"])
        X_transformed = preprocessor.fit_transform(X)

        self. assertIsInstance(X_transformed, np.ndarray)
        self. assertEqual(X_transformed.shape[0], len(self.df))
        self.assertTrue(X_transformed.shape[1] > len(numeric_features))

    def test_no_missing_after_preprocessing(self):
        """Test that preprocessing pipeline removes all missing values."""
        numeric_features = [c for c in self. df.columns if c.startswith("num_")]
        categorical_features = ["cat_small", "cat_med"]

        preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

        X = self.df.drop(columns=["target"])
        X_transformed = preprocessor.fit_transform(X)

        self.assertFalse(np.isnan(X_transformed).any(), "Transformed data should have no NaNs")

    def test_pipeline_consistency(self):
        """Test that pipeline produces same output when fit and transform separately vs fit_transform."""
        numeric_features = [c for c in self. df.columns if c.startswith("num_")]
        categorical_features = ["cat_small", "cat_med"]

        preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

        X = self.df.drop(columns=["target"])

        X_fit_transform = preprocessor.fit_transform(X)

        preprocessor2 = build_preprocessing_pipeline(numeric_features, categorical_features)
        preprocessor2.fit(X)
        X_transform = preprocessor2.transform(X)

        np.testing.assert_array_almost_equal(X_fit_transform, X_transform)


if __name__ == "__main__":
    unittest.main()
