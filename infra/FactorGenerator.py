# Standart libraries
import pandas as pd
import numpy as np

# My libraries
from AssetsData import AssetsData

'''
Input dataframe example:

example_df = pd.DataFrame({'ID'         : ['S0', 'S1', 'S2', 'S0', 'S1', 'S2'],
	                       'F00'        : [   1,    2,    3,    4,    5,   6 ],
	                       'F01'        : [   7,    8,    9,   10,   11,   12],
	                       'TIMESTAMP'  : ['T0', 'T0', 'T0', 'T1', 'T1', 'T1'],
	                       'y'          : [  .1,  -.1,   .5,  -.3,  -.1,    0]}).set_index(['TIMESTAMP', 'ID'])	                       
'''

class FactorGenerator(object):

    def __init__(self, asset_col = 'ID', time_col = 'TIMESTAMP', ret_col = 'y'):
        self.asset_col = asset_col
        self.time_col  = time_col
        self.ret_col   = ret_col
        self.assetdata = AssetsData(asset_col, time_col, ret_col)

    '''
    Return generator of asset specific window dataframe 
    '''
    def get(self, asset_df, row_num, window):
        
        # Get needed factors for each day and information about 'window' previous days
        for index in range(row_num):

            index += 1
            low_bound  = max(index - window, 0)
            high_bound = index

            yield (asset_df.iloc[low_bound:high_bound], asset_df.iloc[index-1].name)

    '''
    Return dataframe with calculated factor
    '''
    def create_factor(self, df, factor_func, factor_name, window, assets = None, save = False, path = '../../data/computed_factors/'):

        # Here we will store indexes and factor values
        index_tuples  = []
        factor_values = []

        # If assets were not given then we take all them
        if assets is None:
            assets = self.assetdata.get_assets_names(df)
        
        # For all the specified assets
        for asset in assets:
        	# Get asset specific dataframe
            asset_df = df[df.index.get_level_values(1) == asset]
            row_num, _ = asset_df.shape
            # Iterate over it and apply the fuction to the specified window
            for (data, index_tuple) in self.get(asset_df, row_num, window):
                
                try:
            	    value = factor_func(data)
                except Exception as e:
                	value = str(e)
                
                factor_values.append(factor_func(data))
                index_tuples.append(index_tuple)

        factor_df = pd.DataFrame( data  = {factor_name : factor_values},
        	                      index = pd.MultiIndex.from_tuples(tuples=index_tuples,
        	                 	                                    names=[self.time_col, self.asset_col]))
        if (save):
        	factor_df.to_csv(path + factor_name + '.csv')

        return factor_df
