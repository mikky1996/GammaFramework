import numpy as np

class AssetsData(object):

    def get_asset_df(self, asset_name, df):
        boolean_vector = (df['ID'] == asset_name)
        return df[boolean_vector]

    def get_asset_df_without_nan(self, asset_name, df):
        return self.get_asset_df(asset_name, df).dropna()
    
    def get_factor_names(self, df):
        return [x for x in df.columns if ((x != 'TIMESTAMP') and (x != 'ID') and (x != 'y'))]

    def get_y_and_factors(self, df):
        col = self.get_factor_names(df) 
        y   = list(df['y'])
        x   = list(np.array(df[col]))
        return y, x  
