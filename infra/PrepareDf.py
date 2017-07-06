import numpy as np
from scipy.stats.mstats import winsorize

# My libraries
from CleanData import CleanData
from AssetsData import AssetsData

class PrepareDf(object):

    def __init__(self):
        self.assetdata = AssetsData()

    # Mikhail style df prep :)
    def rolling_prepare(self, df):
        factors = self.assetdata.get_factor_names(df)
        # Deal with NaN values
        df = CleanData.fill_by_last_valid_observ_for_each_asset(df)
        # Remove outliers (winsorize 1%-99%)
        for factor in factors:
            df = CleanData.apply_to_column_for_each_asset_class_frame_window(df,
                                                                             factor, 
                                                                             CleanData.window_winsorize,
                                                                             window = 30, 
                                                                             min_periods = 1)
        # Normalize factors
        for factor in factors:
            df = CleanData.apply_to_column_for_each_asset_class_frame_window(df,
                                                                             factor,
                                                                             CleanData.window_zscore,
                                                                             window = 30,
                                                                             min_periods = 1)
        return df
    
    # Standart preparation
    def standart_prepare(self, df):
        factors = self.assetdata.get_factor_names(df)
        # Deal with NaN values
        df = df.replace(np.nan, 0)
        # Remove the outliers
        for factor in factors:
            df[factor] = list(winsorize(list(df[factor]), limits=[0.5, 0.5]))
        # Normalize the factors
        for factor in factors:
            df[factor] = (df[factor] - np.mean(df[factor]))/np.std(df[factor])
        return df
