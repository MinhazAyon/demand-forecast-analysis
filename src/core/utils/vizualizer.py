# ---------------------------------------------- Necessary Packages ------------------------------------------

# Core Data Analysis & Manipulation
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

# Visualization Libraries
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

# Typing
from typing import Tuple, List, Union, Callable, Set    # For function/type annotations

# To neatly print a table
from tabulate import tabulate

# Copy Pipeline
from copy import deepcopy

# Import necessary libraries for modeling
from sklearn.pipeline import Pipeline

from core.utils.transformation_techniques import TransformationStrategy
from core.utils.outlier_detector import IQRDetector
from core.utils.io import ensure_directories

# ---------------------------------------------- Helper Functions ------------------------------------------

# Helper class for visualization
class Visualizer:
    """
    A comprehensive EDA class.
    Suggesting finding the best transformation with visuals.
    Helps to visualized the most important features.
    """
    def __init__(self, data: pd.DataFrame):
        self.data= data
        self.iqr_detector = IQRDetector()
        self.transformation = TransformationStrategy
        self.skew_kurt = {}

    @staticmethod
    def modeling_score(skew: float, num_outliers: int, skew_weight=0.8, outlier_weight=0.2) -> float:
        return skew_weight * abs(skew) + outlier_weight * num_outliers

    def recommend_transformation(self, series: pd.Series, is_pos: bool) -> str:
        """
        Recommend the best transformation to apply.
        """
        num_datapoints = len(series)
        results = {}
        if is_pos:

            for method in ['log1p', 'sqrt', 'boxcox', 'yeojohnson', 'original']: # , 'sqrtreflect', 'logreflect'

                try:
                    strategy = self.transformation.registry.get(method)

                    if strategy is None:
                        raise ValueError(f"Transformation method '{method}' is not registered.")
        
                    trans = strategy.apply(series)
                    num_outliers = self.iqr_detector.detect(content= trans, return_length= True)
                    results[method] = {
                    'transformed': trans,
                    'skew_after': trans.skew(),
                    'kurt_after': trans.kurt(),
                    'num_outliers': num_outliers
                    }

                except (ValueError, TypeError) as e:
                    print(f"[{method}] Value/Type Error: {e}")
                except Exception as e:
                    print(f"[{method}] Unexpected error: {e}")
        else:
            for method in ['signedlog1p', 'signedsqrt', 'yeojohnson', 'original']:

                try:
                    strategy = self.transformation.registry.get(method)

                    if strategy is None:
                        raise ValueError(f"Transformation method '{method}' is not registered.")
                    
                    trans = strategy.apply(series)
                    num_outliers = self.iqr_detector.detect(content= trans, return_length= True)
                    results[method] = {
                    'transformed': trans,
                    'skew_after': trans.skew(),
                    'kurt_after': trans.kurt(),
                    'num_outliers': num_outliers
                    }

                except (ValueError, TypeError) as e:
                    print(f"[{method}] Value/Type Error: {e}")
                except Exception as e:
                    print(f"[{method}] Unexpected error: {e}")

        # --- Choose the best one ---
        best_method = min(
            results,
            key=lambda k: Visualizer.modeling_score(
                results[k]['skew_after'], 
                results[k]['num_outliers'] / num_datapoints
            )
        )
        
        for key in list(results.keys()):
            if  key == best_method:
                results[f'{key}*'] = results.pop(key)

        return results

    @staticmethod
    def checkpos(series: pd.Series):
        if (series < 0).any():
            return False
        else:
            return True
    
    @staticmethod
    def load_update_dict(dir_path, filename, new_data):

        # Ensure directory exists
        ensure_directories(dir_path)

        dir_path = Path(dir_path)
        
        file_path = dir_path / filename
        
        # Load existing dictionary if file exists, else start with empty dict
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update the dictionary with new_data
        data.update(new_data)
        
        # Save the updated dictionary back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        return data
        
    @staticmethod
    def box_plot(series: pd.Series, tfont: int = 12, ax: Axes = None, add_title: str = None,
                 color: str= 'lightgreen', flierprops: Set = {'markersize': 4, 'markeredgewidth': 0.5}):
        
        if series is None:
            raise ValueError("You must provide `Series` for box plot.")
        
        if ax == None:
            # Figure size
            plt.figure(figsize=size)

        var_name= series.name.title().replace('_', ' ')

        sns.boxplot(
            x= series,
            color= color,
            flierprops= flierprops,
            ax= ax
        )

        if ax == None:
            plt.title(f'{var_name}', fontsize= tfont)
            plt.xlabel(f'{var_name}', fontsize= tfont)

        else:
            ax.set_title(f'{var_name} ({add_title})', fontsize= tfont)
            ax.set_xlabel(f'{var_name}', fontsize= tfont)
        
    @staticmethod
    def hist_plot(series: pd.Series, tfont: int = 12, ax: Axes = None, add_title: str = None,
                  size: Tuple = (8,5), color: str= 'skyblue', linewidth: float = 1.0):
        
        if series is None:
            raise ValueError("You must provide `Series` for histogram.")
        
        if ax == None:
            # Figure size
            plt.figure(figsize=size)

        var_name= series.name.title().replace('_', ' ')
        
        # Change KDE line thickness
        def kdelinewidth(plot, linewidth = linewidth):
            for line in plot.lines:
                line.set_linewidth(linewidth)
    
        # Original data distribution (Histplot)
        plot = sns.histplot(
            series, 
            kde=True, 
            bins=50, 
            color=color,
            ax= ax
        )

        if ax == None:
            kdelinewidth(plot)
            plt.title(f'{var_name}', fontsize= tfont)
            plt.xlabel(f'{var_name}', fontsize= tfont)
            plt.ylabel('Frequency', fontsize= tfont)
            plt.xlim(series.min() - 1, series.max() + 1)
        else:
            kdelinewidth(plot)
            ax.set_title(f'{var_name} ({add_title})', fontsize= tfont)
            ax.set_xlabel(f'{var_name}', fontsize= tfont)
            ax.set_ylabel('Frequency', fontsize= tfont)
            ax.set_xlim(series.min() - 1, series.max() + 1)

    # Function to visualize numerical distribution
    def numplot(
            self, 
            variable_name: str,
            tfont: int= 10,
            size: Tuple= (12, 4),
            save_dir: str = False,
            change_technique: str = None
            ) -> Tuple[Figure, str]:

        series= self.data[variable_name]
        is_pos = self.checkpos(series)

        result= self.recommend_transformation(series, is_pos)

        method = [key for key in result.keys() if '*' in key][0]
        transformed_series = result[method]['transformed']

        # Set the plots structure
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=size)
    
        # Original data distribution (Histplot)
        self.hist_plot(series=series, tfont = tfont, ax = ax[0], add_title= "Original")

        # Original data distribution (Boxplot)
        self.box_plot(series=series, tfont = tfont, ax=ax[1], add_title= "Original")

        # Recommended Transformation
        self.hist_plot(series=transformed_series, tfont=tfont, ax=ax[2], add_title=f"{method.split('*')[0]}")
    
        # Recommended Transformation
        self.box_plot(series=transformed_series, tfont=tfont, ax=ax[3], add_title=f"{method.split('*')[0]}")
    
        print(f"Skewness and Kurtosis: {series.name.title().replace('_', ' ')}")

        keys_to_remove = ['transformed', 'num_outliers']
        for values in result.values():
            for key in keys_to_remove:
                values.pop(key, None)

        table= pd.DataFrame.from_dict(result, orient='index').reset_index().rename(columns={
            'index': 'Method',
            'skew_after': 'Skewness',
            'kurt_after': 'Kurtosis'
            }).sort_values(by='Skewness')
        
        # Neatly print skewness kurtosis
        print(tabulate(table.values.tolist(), headers=table.columns, tablefmt="fancy_grid"))

        # Show Plots
        plt.tight_layout()
        plt.show()

        if save_dir:
            if change_technique == None:
                dir_path = Path('../notebooks/models/transformers')
                Visualizer.load_update_dict(
                    dir_path= dir_path,
                    filename= 'transformation_list.json',
                    new_data= {f'{variable_name}':f'{method}'}
                )
            else:
                dir_path = Path('../notebooks/models/transformers')
                Visualizer.load_update_dict(
                    dir_path= dir_path,
                    filename= 'transformation_list.json',
                    new_data= {f'{variable_name}':f'{change_technique}'}
                )