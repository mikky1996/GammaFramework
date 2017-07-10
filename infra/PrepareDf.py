import numpy as np
from scipy.stats.mstats import winsorize

# My libraries
from CleanData import CleanData
from AssetsData import AssetsData

class PrepareDf(object):

    def __init__(self):
        self.assetdata = AssetsData()
        self.cleandata = CleanData()

    '''
    Realistic approach of data preparation
    '''
    def rolling_prepare(self, df, window = 30, min_periods = 1):

        factors = self.assetdata.get_factor_names(df)
        # Deal with NaN values
        df = self.cleandata.fill_by_last_valid_observ_for_each_asset(df)
        # Remove outliers (winsorize 1%-99%)
        for factor in factors:
            df = self.cleandata.apply_to_column_for_each_asset_class_frame_window(df,
                                                                                  factor, 
                                                                                  CleanData.window_winsorize,
                                                                                  window      = window, 
                                                                                  min_periods = min_periods)
        # Normalize factors
        for factor in factors:
            df = self.cleandata.apply_to_column_for_each_asset_class_frame_window(df,
                                                                                  factor,
                                                                                  CleanData.window_zscore,
                                                                                  window      = window,
                                                                                  min_periods = min_periods)
        return df

    '''
    Standart preparation
    '''
    def standart_prepare(self, df, limits = [0.05, 0.05]):
        factors = self.assetdata.get_factor_names(df)
        # Deal with NaN values
        df = df.replace(np.nan, 0)
        # Remove the outliers
        for factor in factors:
            df[factor] = list(winsorize(list(df[factor]), limits=limits))
        # Normalize the factors
        for factor in factors:
            if (np.std(df[factor]) != 0):
                df[factor] = (df[factor] - np.mean(df[factor]))/np.std(df[factor])
            else:
                df[factor] = (df[factor] - np.mean(df[factor]))
        return df
