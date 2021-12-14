import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ReduceMemoryUsage(BaseEstimator, TransformerMixin):

    def __init__(self,X):
        self.X = X

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        df = X.copy()

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        return df


class ColumnAggregation(BaseEstimator, TransformerMixin):

    def __init__(self,name_new_column, start_with):
        self.name_new_column = name_new_column
        self.start_with = start_with

    def fit(self, X,y=None):
        return self

    def transform(self, X):
        df = X.copy()

        column_to_aggregate = [x for x in df.columns if x.startswith(self.start_with)]
        df[self.name_new_column] = df[column_to_aggregate].sum(axis=1)

        return df

class CapAngularValues(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df[self.column][df[self.column] < 0] += 360
        df[self.column][df[self.column] > 359] -= 360

        return df

class ManhattanDistance(BaseEstimator,TransformerMixin):

    def __init__(self, column):
        self.column = column


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df[self.column] =[abs(x) for x in df[self.column]]

        return df


class RemoveUnusefulFeature(BaseEstimator, TransformerMixin):

    def __init__(self, column, axis=1):
        self.column = column
        self.axis = axis


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df.drop([self.column], self.axis)
        return df
