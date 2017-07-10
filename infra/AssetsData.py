import numpy as np

class AssetsData(object):

    def __init__(self, asset_col = 'ID', time_col = 'TIMESTAMP', ret_col = 'y'):
        self.asset_col = asset_col
        self.time_col  = time_col
        self.ret_col   = ret_col

    '''
    Get asset specific dataframe
    '''
    def get_asset_df(self, asset_name, df):
        boolean_vector = (df.index.get_level_values(1) == asset_name)
        return df[boolean_vector]

    '''
    Get asset specific dataframe, where 
    '''
    def get_asset_df_without_nan(self, asset_name, df):
        return self.get_asset_df(asset_name, df).dropna()
    
    '''
    Get names of assets of securities out of the dataframe
    '''
    def get_assets_names(self, df):
        list_of_names = list(set(list(df.index.get_level_values(1))))
        list_of_names.sort()
        return list_of_names
    
    '''
    Get factors of the dataframe
    '''
    def get_factor_names(self, df):
        return [x for x in df.columns if ((x != self.time_col) and (x != self.asset_col) and (x != self.ret_col))]

    '''
    Get X and Y for regression out of dataframe
    '''
    def get_x_y(self, df, factors = None):

        # If col is not defined, return all the factors
        if factors is None:
            factors = self.get_factor_names(df)

        # Define and return x and y
        y   = list(df[self.ret_col])
        x   = list(np.array(df[factors]))
        return x, y
