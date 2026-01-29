# core/utils/general_analyzer.py

# --- Standard Python imports ---
import logging
import os
import numpy as np
from pathlib import Path
import re
from typing import Optional, Union


# --- Third-party imports ---
from collections import Counter 
import pandas as pd
from tabulate import tabulate

# --- Local imports ---
from core.utils.file_manager import ReadManager, WriteManager
from core.utils.logger import get_logger


log = get_logger(__name__)
read = ReadManager()
write = WriteManager()

class GeneralAnalyzer:
    """
    A class to handle initial data processing like object to catgeory, find strings in numeric,
    numeric to categories etc.

    Attributes:
        log (logging.Logger): Logger instance for logging events.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes the GeneralAnalyzer object with an optional logger and dataframe.
        
        Args:
            logger (Optional[logging.Logger]): A logger to be used by the strategy (default None).
        """
        base = logger or log
        self.log =  base.getChild(self.__class__.__name__)

    def data_info(self, data: pd.DataFrame) -> pd.DataFrame: 
        """
        Give general informations about data in tabular format.

        Args:
            data (pd.DataFrame): The dataset to get info.

        Returns:
            pd.DataFrame: A dataframe with genaral informations.

        """
        num_rows = data.shape[0]                                              # total number of observations
        num_col = data.shape[1]                                               # total number od columns/ features
        num_dtype = len(set([x.lower() for x in data.dtypes.astype(str)]))    # total number of data types
        count = dict(Counter([x.lower() for x in data.dtypes.astype(str)]))   # counting the number of occurennces of data type
        num_duplicates = int(data.duplicated().sum())                         # total number of duplicates
        num_null = data[data.isna().any(axis=1)].shape[0]                  # total NAN values
 
        if num_duplicates == 0:
            duplicated = 'No duplicated rows.'
        else: 
            duplicated = f'{num_duplicates} number of duplicated rows.'

        if num_null == 0:
            nulled = 'No NAN values.'
        else: 
            nulled = f'{num_null} number of NAN values.'

        table = pd.DataFrame({
            'Variables': [
                'Rows', 'Columns', 'Dtypes', 'Duplicates', 'NAN'
                ],
            'Information': [
                f'{num_rows} number of rows',
                f'{num_col} number of columns.',
                f'{num_dtype} unique data types. {count}.',
                duplicated,
                nulled
                ]

        })

        return print(tabulate(table.values.tolist(), headers=table.columns, tablefmt="fancy_grid"))
    
    def convert_lower_case_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert column name to lower case.

        Args:
            df (pd.DataFrame): The dataframe to process.

        Return:
            pd.DataFrame: Dataset with lower case column names.
        """
        for cols in data.columns:
            data.rename(
                columns= {
                    cols: cols.lower()
                },
                inplace= True
            )
        
        self.log.info("lower-case convertion.success")
        return data
    
    def object_cat(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        To convert object data type to catgegory.

        Args:
            data (pd.DataFrame): The dataset to process.

        Returns:
            pd.Dataframe: Dataframe with category data types.
        """
        categorical_cols = data.select_dtypes(include=['object']).columns

        self.log.info("convertion.begin columns=%s", categorical_cols)

        for col in categorical_cols:
            data[col] = data[col].astype('category')

        self.log.info("convertion.success")

        return data

    def find_strings_in_numeric(self, data: pd.DataFrame, columns: list[str] = None) -> dict:
        """
        Finds and returns strings that contain both digits and non-digit characters (e.g., '600 mile', '120mph') 
        in object-type columns.

        Args:
            data (pd.DataFrame): The dataset to process.
            columns (list[str]): Specific columns to check. If None, all object dtype columns are checked.

        Returns:
            dict: Dictionary mapping column names to a list of unique strings that include both numbers and text.
        """
        pattern = re.compile(r'\d+\s*[a-zA-Z]+')  # e.g., 600mile, 600 mile, 120mph
        result = {}
 
        if columns is None:
            self.log.info(f"Column list is None, checking in all Object type columns")
            columns = data.select_dtypes(include='object').columns.tolist()

        for col in columns:
            values = data[col].dropna().astype(str)
            matches = values.apply(lambda x: bool(pattern.search(x)))
            unit_strings = values[matches].unique().tolist()
            if unit_strings:
                result[col] = unit_strings

        return result
    
    def clean_numeric(self, data: Union[pd.DataFrame, dict], columns: list[str] = None) -> Union[pd.DataFrame, dict]:
        """
        Cleans numeric and boolean values in a DataFrame or a dict.
        - For DataFrame: cleans specified columns (removes non-numeric characters, converts to float).
        - For dict: cleans numeric strings and boolean strings, keeps non-numeric values as-is.

        Args:
            data (pd.DataFrame or dict): Input data.
            columns (list[str], optional): Columns to clean (required for DataFrame).

        Returns:
            pd.DataFrame or dict: Cleaned data.
        """

        def _convert_value(v):
            if isinstance(v, str):
                # Handle boolean strings
                if v.lower() == "true":
                    return True
                elif v.lower() == "false":
                    return False

                # Clean numeric string
                clean_v = re.sub(r'[^0-9eE\.\-]+', '', v)
                try:
                    if '.' in clean_v or 'e' in clean_v.lower():
                        return float(clean_v)
                    else:
                        return int(clean_v)
                except ValueError:
                    return v
            return v

        if isinstance(data, pd.DataFrame):
            if columns is None:
                raise ValueError("For DataFrame input, 'columns' must be provided.")
            df = data.copy()
            for col in columns:
                df[col] = df[col].apply(
                    lambda x: float(re.sub(r'[^\d\.]+', '', str(x))) if pd.notnull(x) else np.nan
                )
            self.log.info(f"Successfully cleaned numeric columns: {columns}")
            return df

        elif isinstance(data, dict):
            cleaned = {k: _convert_value(v) for k, v in data.items()}
            self.log.info("Successfully cleaned numeric values in dict.")
            return cleaned

        else:
            raise TypeError("Input must be a pandas DataFrame or a dict.")
    
    def replace_to_category(self, data: pd.DataFrame, column: str | pd.Series, limit: int, replace_by: str, 
                            save_path: str, change_value: bool = False) -> pd.DataFrame:
        """
        Make new categories by limiting the counts of rare values in a series, or by 
        changing specific categories based on a dictionary saved at the provided path.

        Args:
            data (pd.DataFrame): The pandas DataFrame to process.
            column (str): Column to process.
            limit (int): The threshold count below which values in the series will be categorized as rare.
            replace_by (str): The value that will replace the rare categories in the series.

            save_path (str): 
                The path to the dictionary (JSON or similar format) where the category mappings are saved.

            change_value (bool, optional): 
                Whether to replace values in the original DataFrame with `replace_by` (default to False).

        Returns:
            pd.DataFrame: The updated DataFrame with replaced categories in the specified column.

        Notes:
            If `change_value` is set to True, the method will update both the original dataframe and the saved 
            dictionary with the changes. Otherwise, it will only update the dictionary and replace the rare categories 
            in the dataframe with `replace_by`.
        """
        if isinstance(column, str):
            series = data[column]
        else:
            series = column

        save_path = Path(save_path)

        if os.path.exists(save_path):
            self.log.info(f"Reading dictionary from {save_path}")
            list_dic = read.files(
                input_path= save_path
            )
        else:
            self.log.info(f"No dictionary found at {save_path}, starting with an empty dictionary.")
            list_dic = {}

        category_storage = {}
        value_counts = series.value_counts()
        rare = value_counts[value_counts < limit].index

        if change_value:
            self.log.info(f"Replacing rare categories with '{replace_by}' in the original dataframe.")
            for val in rare:
                keys_to_change = [key for key, values in list_dic.items() if values == val]
                for k in keys_to_change:
                    list_dic[k] = replace_by

                    # Change in original dataframe
                    data[series.name] = data[series.name].replace(val, replace_by)
        else:
            self.log.info(f"Updating dictionary with new categories for rare values.")
            for cat in rare:
                category_storage[cat] =  replace_by

            # Update the dictionary with new_data
            list_dic.update(category_storage)
            # Change in original dataframe
            data[series.name] = data[series.name].replace(rare, replace_by)

        # Save Dictionary
        self.log.info(f"Saving updated dictionary to {save_path}")
        write.files(
            file_content= list_dic,
            output_path= save_path
        )

        return data



