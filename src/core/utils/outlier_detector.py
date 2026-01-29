from abc import ABC, abstractmethod
import logging
import pandas as pd
from typing import Optional, Union, Tuple, Literal

from core.utils.logger import get_logger

log = get_logger(__name__)

# -----------------------------
# | Base Outlier Detector     |
# -----------------------------
class BaseOutlierDetector(ABC):
    """
    A detector that detect outliers in data.

    Attributes:
        log (logging.Logger): Logger instance for logging events.
    """
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initializes the BaseOutlierDetector.

        Args:
            log (logging.Logger): Logger instance for logging events.
        """
        base = logger or log
        self.log =  base.getChild(self.__class__.__name__)

    @abstractmethod
    def detect(
        self, 
        content: pd.DataFrame | pd.Series, 
        column: str = None,
        return_length: bool = False
        ) -> Union[Tuple[pd.DataFrame, float, float], int]:
        """
        Abstract method to detect outliers in a series of dataset.
        
        Args:
            content (pd.DataFrame | pd.Series): The dataset/ series to detect outliers.
            column (str): Name of the column to detect outliers (default None).
            return_length (bool): Will return only the count of outliers (default False)

        Returns:
            Union[Tuple[pd.DataFrame, float, float], int]:
                Either a tuple (outliers DataFrame, lower bound, upper bound) or the count of outliers.
        """
        ...

# -----------------------------
# | IQR-Based Outlier Detector |
# -----------------------------
class IQRDetector(BaseOutlierDetector):
    """
    Concrete implementation of the BaseOutlierDetector for detecting outliers.
    """
    def detect(self, content: pd.DataFrame | pd.Series, column: str = None, return_length: bool = False) -> Union[Tuple[pd.DataFrame, float, float], int]:
        """
        Detect outliers in a series of dataset.
        
        Args:
            content (pd.DataFrame | pd.Series): The dataset/ series to detect outliers.
            column (str): Name of the column to detect outliers.
            return_length (bool): Will return only the count of outliers (default False)

        Returns:
            Union[Tuple[pd.DataFrame, float, float], int]:
                Either a tuple (outliers DataFrame, lower bound, upper bound) or the count of outliers.
        """
        if isinstance(content, pd.Series):
            series = content
            return_length = True
        else:
            series = content[column]
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        self.log.info(
            "Detecting outliers: Q1=%.3f, Q3=%.3f, IQR=%.3f, lower=%.3f, upper=%.3f",
            Q1, Q3, IQR, lower, upper
        )

        if return_length:
            count = ((series < lower) | (series > upper)).sum()
            self.log.info("Has %d outliers based on IQR method", count)
            return count
        else:
            outliers = content[(series < lower) | (series > upper)]
            self.log.info("Column '%s': found %d outlier rows", column, len(outliers))
            return outliers, lower, upper