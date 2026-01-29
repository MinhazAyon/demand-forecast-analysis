# core/utils/inverse_transformed_regressor.py

# --- Standard Python imports ---
import logging
import os
from pathlib import Path
from typing import Callable, Optional

# --- Third-party imports ---
from sklearn.base import BaseEstimator, clone
import pandas as pd
import numpy as np

# --- Local imports ---
from core.utils.file_manager import ReadManager, WriteManager
from core.utils.transformation_techniques import TransformationStrategy
from core.utils.io import find_path

# Initialized Classes
read = ReadManager()
write = WriteManager()

class InverseTransformedRegressor(BaseEstimator):
    """
    A transformer that applies inverse transformations to predictions and true values.

    Attributes:
        log (logging.Logger): Logger instance for logging events.
        base_model_ (BaseEstimator): Cloned and fitted instance of the base model.
        target (str): Name of the target variable.
        transformer_dir (str): Directory containing the transformer files.
        metrics (callable): Metric function used for scoring predictions.
        greater_is_better (bool): Indicates if higher metric values are better.
        
        transformation (TransformationStrategy): 
            Registry holding transformation and inverse transformation functions.

        method (str): Name of the transformation method applied to the target (e.g., 'boxcox', 'yeojohnson').
        transformer (object or None): Loaded transformer object if required for the method; otherwise None.
    """
    def __init__(
        self, 
        base_model: Optional[BaseEstimator] = None,         
        target: str = None, 
        transformer_dir: str = None,
        metrics: Optional[Callable] = None,
        is_final: bool = False,
        greater_is_better: bool = True,
    ):
        """
        Initializes the InverseTransformedRegressor.

        Args:
            base_model (Optional[BaseEstimator]): The base estimator to fit and predict with transformed targets.
            target (str): The name of the target variable to transform.
            metrics ( Optional[Callable]): The metric function to evaluate predictions (default None).
            greater_is_better (bool): Whether higher metric values are better (True) or worse (False) (default True).
            transformation_list_path (str): Path to the JSON file containing the transformation list.
            transformer_dir (str): Directory containing the transformer files.
        """
        self.base_model= base_model
        self.target= target
        self.transformer_dir = transformer_dir
        self.metrics= metrics
        self.greater_is_better= greater_is_better
        self.transformation= TransformationStrategy
        self.is_final = is_final
        
        if self.is_final:
            self.filename = target + "_final"
        else:
            self.filename = target

        # Load transformation method
        transformation_list = read.files(
            input_path= Path(self.transformer_dir) / "transformation_list.json"
        )
        self.method = transformation_list[self.target].split('*')[0]

        if self.method in ['boxcox', 'yeojohnson']:
            self.transformer = read.files(
                input_path = find_path(filename= self.filename, directory= transformer_dir)
            )

        else:
            self.transformer = None

    def fit(self, X, y): 
        """
        To fit the base model on the transformed target variable.
        """
        # Fit the base model on the transformed target y
        self.base_model_ = clone(self.base_model)
        self.base_model_.fit(X, y)
        return self
    
    def transform(self, series):
        """
        Apply the inverse transformation to a series of values.
        """
        # Inverse transform predictions to original scale
        transformed_series = self.transformation.registry.get(self.method).inverse(
            series = series, 
            transformer = self.transformer
        )
        return transformed_series

    def predict(self, X):
        """
        Predict target values in the original scale.
        """
        y_pred_transformed = self.base_model_.predict(X)      # Predict in transformed space 
        y_pred= self.transform(pd.Series(y_pred_transformed)) # Inverse transform predictions to original scale
        return y_pred
    
    def score(self, X, y, sample_weight=None):
        """
        Compute a score using the metric provided, with predictions and true values
        on the original scale.
        """
        try:
            y_pred = self.predict(X)     # Predicted values
            y_true = self.transform(y)  # Inverse transform true values to original scale

            if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
                return float("inf")
            
            score = self.metrics(
                y_true, y_pred
            )

            # In sklearn, higher is better — if needed, negate
            return score if self.greater_is_better else -score
        
        except Exception as e:
            return float("inf")
    