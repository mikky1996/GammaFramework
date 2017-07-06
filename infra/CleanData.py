# Standart libraries
import pandas as pd
import numpy as np
import scipy
import warnings

# My libraries
import AssetsData

class CleanData(object):

#--------------------------< Wrappers for NaN elimination > --------------------------

    # This function will find the previous valid observation for specific asset and replace the NaN by it
    def fill_by_last_valid_observ_for_each_asset(df):
        # If the NaN is in the first valid observation for specific asset then it will be replaced with 0
        df['ID1'] = df['ID']
        df = df.groupby('ID').fillna(method = 'ffill').replace(np.nan, 0)
        df['ID'] = df['ID1']
        del df['ID1']
        return df

    # This function will find the last valid observation for any asset and replace the NaN by it
    def fill_by_last_valid_observ_for_all_asset(df):
        return df.fillna(method = 'ffill').replace(np.nan, 0)

    # This functino will just drop all the rows with NaN values
    def drop_all(df):
        return df.dropna()

#----------------------< Wrappers for feature transformation >------------------------

    # This function is equvivalent to:
    # 1) For each asset class, create dataframe, with dates and the asset as index and factors
    # 2) Get out of it the specified factor
    # 3) Go over this column and apply "func" to window of fields and substitute by the result value the value on the end of the window
    # 4) After finished with one asset class go to the next one, if no more touched asset classes exist, then...
    # 5) ... substitute the existed column by the new one and return dataframe
    # !!! It starts 'func' only if the element on edge is not NaN !!! 
    # (Actually it's behaviour is weird when there are any of NaN values, so better to exclude them before start this function)
    def apply_to_column_for_each_asset_class_frame_window(df, column_name, func, window = 30, min_periods = 1):

        warnings.warn("This function ignores NaN", DeprecationWarning)

        col = df.groupby('ID', group_keys = False)[column_name].rolling(window=window, min_periods=min_periods).apply(func).sort_index()
        real_index = [pair[1] for pair in col.index.values]
        col.index = real_index
        df[column_name] = col
        return df
    
    def apply_for_each_asset_class_frame_window(df, func, window = 30, min_periods = 1):

        warnings.warn("This function ignores NaN", DeprecationWarning)

        col = df.groupby('ID', group_keys = False).rolling(window=window, min_periods=min_periods).apply(func).sort_index()
        real_index = [pair[1] for pair in col.index.values]
        col.index = real_index
        df[column_name] = col
        return df

    # This function applies 'func' to the whole factor column (by windows) with out division on asset classes
    def apply_for_all_asset_class_frame_window(df, column_name, func, window = 30, min_periods = 1):
        
        warnings.warn("This function ignores NaN", DeprecationWarning)

        col = df[column_name].rolling(window=window, min_periods=min_periods).apply(func).sort_index()
        df[column_name] = col
        return df

#--------------------------< Functions for feature transformation >---------------------  

    # Function for window z-score, pass it to < Wrappers for feature transformation > functions
    def window_zscore(s):
        if (len(s) == 1):
            return s[-1]
        if (np.std(s) == 0):
            return 0
        return float(s[-1] - np.mean(s))/np.std(s)

    # Function function detects outliers and if it finds it replases it with median of the window
    def window_winsorize(s):
        q = pd.Series(data = s).quantile([0.01, 0.99])
        q_low  = q.iloc[0]
        q_high = q.iloc[1]
        s = list(s)
        if ((s[-1] < q_low) or (s[-1] > q_high)):
            if (np.median(s) is None):
                print(s)
            return np.median(s)
        if (s[-1] is None):
            print(s)
        return s[-1]
