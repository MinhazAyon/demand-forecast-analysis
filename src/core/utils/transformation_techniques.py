from abc import ABC, abstractmethod
from typing import Optional
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pandas as pd


# -----------------------------------
# | Base Transformation Strategy    |
# -----------------------------------
class TransformationStrategy(ABC):
    """
    Abstract base class for transformation strategies.
    Enforces all subclasses to implement the 'apply' method.
    Automatically registers strategies by name.
    """
    registry = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers each subclass using the class name.
        """
        super().__init_subclass__(**kwargs)
        name = cls.__name__.replace("Transform", "").lower()
        TransformationStrategy.registry[name] = cls()

    @abstractmethod
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        """
        Apply the transformation to the input series.

        Args:
            series (pd.Series): Input numerical series.

        Returns:
            pd.Series: Transformed series.
        """
        pass

    @abstractmethod
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        """
        Reverse the transformation using the fitted transformer.
        """
        pass

# -------------------------------
# | Concrete Transformations    |
# -------------------------------
class Log1pTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for log transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply log1p transformation."
        cleaned = series.dropna()
        transformed = np.log1p(cleaned)
        return transformed.reindex(series.index, fill_value=np.nan)
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of log1p transformation."
        cleaned = series.dropna()
        inversed = np.expm1(cleaned)
        return inversed.reindex(series.index, fill_value=np.nan)


class SqrtTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for square root transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply square root transformation."
        cleaned = series.dropna()
        transformed = np.sqrt(cleaned)
        return transformed.reindex(series.index, fill_value=np.nan)
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of square root transformation."
        cleaned = series.dropna()
        inversed = np.sqrt(cleaned)
        return inversed.reindex(series.index, fill_value=np.nan)

class SqrtReflectTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for square root reflect transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply square root reflect transformation."
        cleaned = series.dropna()
        transformed = np.sqrt(cleaned .max() + 1 - cleaned)
        return transformed.reindex(series.index, fill_value=np.nan)
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of square root reflect transformation."
        cleaned = series.dropna()
        inversed = self._max_val + 1 - np.square(cleaned)
        return inversed.reindex(series.index, fill_value=np.nan)


class LogReflectTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for log reflect transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply log reflect transformation."
        cleaned = series.dropna()
        transformed = np.log1p(cleaned .max() + 1 - cleaned)
        return transformed.reindex(series.index, fill_value=np.nan)
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of log reflect transformation."
        cleaned = series.dropna()
        inversed = self._max_val + 1 - np.expm1(cleaned)
        return inversed.reindex(series.index, fill_value=np.nan)


class SignedLog1pTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for signed log transform to handle negative values.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply signed log1p transformation."
        cleaned = series.dropna()
        transformed = np.log1p(np.abs(cleaned)) * np.sign(cleaned)
        return transformed.reindex(series.index, fill_value=np.nan)
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of signed log1p transformation."
        cleaned = series.dropna()
        inversed = np.expm1(np.abs(cleaned)) * np.sign(cleaned)
        return inversed.reindex(series.index, fill_value=np.nan)


class SignedSqrtTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for signed square root transform to handle negative values.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply signed square root transformation."
        cleaned = series.dropna()
        transformed = np.sqrt(np.abs(cleaned)) * np.sign(cleaned)
        return transformed.reindex(series.index, fill_value=np.nan)
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of signed square root transformation."
        cleaned = series.dropna()
        inversed = np.square(cleaned) * np.sign(cleaned)
        return inversed.reindex(series.index, fill_value=np.nan)
    
class BoxCoxTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for boxcox transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply boxcox transformation."
        cleaned = series.dropna()
        transformer= PowerTransformer(method='box-cox', standardize=False)
        transformed = transformer.fit_transform(cleaned.values.reshape(-1, 1)).flatten()
        result = pd.Series(transformed, index=cleaned.index).reindex(series.index, fill_value=np.nan)
        result.name = series.name
        return (result, transformer) if return_transformer else result
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of boxcox transformation."
        cleaned = series.dropna()
        inversed = transformer.inverse_transform(cleaned.values.reshape(-1, 1)).flatten()
        result = pd.Series(inversed, index=cleaned.index).reindex(series.index, fill_value=np.nan)
        result.name = series.name
        return result
    
class YeoJohnsonTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for Yeo Johnson transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        "Apply Yeo Johnson transformation."
        cleaned = series.dropna()
        transformer= PowerTransformer(method='yeo-johnson', standardize=False)
        transformed = transformer.fit_transform(cleaned.values.reshape(-1, 1)).flatten()
        result = pd.Series(transformed, index=cleaned.index).reindex(series.index, fill_value=np.nan)
        result.name = series.name
        return (result, transformer) if return_transformer else result 
    
    def inverse(self, series: pd.Series, transformer: Optional[object] = None) -> pd.Series:
        "Inverse of Yeo Johnson transformation."
        cleaned = series.dropna()
        inversed = transformer.inverse_transform(cleaned.values.reshape(-1, 1)).flatten()
        result = pd.Series(inversed, index=cleaned.index).reindex(series.index, fill_value=np.nan)
        result.name = series.name
        return result
    
class OriginalTransform(TransformationStrategy):
    """
    Concrete implementation of the TransformationStrategy for no transform.
    """
    def apply(self, series: pd.Series, return_transformer: bool = False) -> pd.Series:
        return series
