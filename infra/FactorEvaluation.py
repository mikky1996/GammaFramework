import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

class FactorEvaluation(object):

    def __init__(self, asset_col = 'ID', time_col = 'TIMESTAMP', ret_col = 'y', pred_col = 'y_pred'):
        self.asset_col = asset_col
        self.time_col  = time_col
        self.ret_col   = ret_col
        self.pred_col  = pred_col

    '''
    IC_spearmanr takes as input DF with first indexing by time and second as assets
    IC_spearmanr returns DF with dayli spearman correlation between predictions and real returns
    '''
    def IC_spearmanr(self, df):
        return df.groupby(df.index.get_level_values(0)).apply(lambda df: spearmanr(df[self.ret_col], df[self.pred_col])[0])
    
    '''
    IC_pearsonr takes as input DF with first indexing by time and second as assets
    IC_pearsonr returns DF with dayli pearsonr correlation between predictions and real returns
    '''
    def IC_pearsonr(self, df):
    	return df.groupby(df.index.get_level_values(0)).apply(lambda df: pearsonr(df[self.ret_col], df[self.pred_col])[0])

    '''
    alphalens_data takes the usuall DF with numerical indexing and time column
    Returns data which can be fed into alphalens utils function to get the final df to feed into plot generator
    '''
    
    def alphalens_data(self, df, start_date = '1/1/2011', time_zone = 'UTC'):
        
        d_with_col = df.reset_index(level=[self.time_col, self.asset_col])

        # Create time list with unique names of the timestamp from df and the timeseries
        rng = pd.date_range(start_date, periods=len(set(d_with_col[self.time_col])), freq='D')
        time_list = list(set(d_with_col[self.time_col]))
        time_list.sort()

        # Map Timestamp values to some FAKE real dates to work after with a helper library
        time_dict = {}
        for i, ts in enumerate(time_list):
            time_dict[ts] = rng[i]

        # Create new time stamp column and add it to dataframe
        new_time = []
        for ts in d_with_col[self.time_col]:
            new_time.append(time_dict[ts])
        
        # Substitute the column with new time
        d_with_col[self.time_col] = new_time

        # Prepare data
        factor = (d_with_col.set_index([self.time_col, self.asset_col])).tz_localize(time_zone,level = 0)
        pricing = (d_with_col.pivot(index=self.time_col, columns=self.asset_col, values=self.ret_col)).tz_localize(time_zone,level = 0)

        return factor, pricing
