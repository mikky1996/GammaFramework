import numpy as np

class Asset(object):
    
    def __init__(self, name, asset_col = 'ID', time_col = 'TIMESTAMP', ret_col = 'y'):
        self.name        = name
        self.factors     = None
        self.models      = {}
        self.factor_stat = {}
        self.day_count   = 0
        self.ret_col     = ret_col
        self.accepted    = True
        self.getdata     = GetData(asset_col, time_col, ret_col)

    def select_factors(self, df_train, std_threshold_min = 0.001, std_threshold_max = 300, exclude_factors = []):

        asset_df = df_train[df_train.index.get_level_values(1) == self.name]

        if  asset_df.empty:
            self.accepted = False

            return

        all_factors      = self.getdata.get_factor_names(asset_df)
        selected_factors = []

        for factor in all_factors:
            std = np.std(asset_df[factor])
            if ((std > std_threshold_min) and (std < std_threshold_max) and (factor not in exclude_factors)):
                selected_factors.append(factor)

        self.factors = selected_factors
        return

    def config_factor_stat(self, df_train, stats = {'mean' : np.mean, 'var' : np.var, 'std': np.std}):

        # If no factors specified then, just return
        if (self.factors is None):
            return

        asset_df = df_train[df_train.index.get_level_values(1) == self.name]

        for factor in self.factors + [self.ret_col]:

            local_stat = {}

            for key in stats.keys():
                local_stat[key] = stats[key](asset_df[factor])

            self.factor_stat[factor] = local_stat
        return

    def filter_asset(self, func, *args, **kwargs):

        if (func(self, *args, **kwargs)):
            self.accepted = False
        else:
            self.accepted = True        # Default
        return

    def train(self, df_train, model, model_name):

        if (not self.accepted):
            self.models[model_name] = None
            return

        asset_df_train = df_train[df_train.index.get_level_values(1) == self.name].copy(deep=True)

        # Prepare df_train
        for factor in self.factors:
            asset_df_train.loc[:, (factor)] = (asset_df_train[factor] - self.factor_stat[factor]['mean'])/self.factor_stat[factor]['std']

        # Train model
        model.fit(asset_df_train[self.factors], asset_df_train[self.ret_col])

        # Save the model
        self.models[model_name] = model
        return

    def predict(self, df_test, model_name, pred_col = 'y_pred', filter_pred = None, *args, **kwargs):

        asset_df_test = df_test[df_test.index.get_level_values(1) == self.name].copy(deep=True)

        # If no such asset is in the test set
        if asset_df_test.empty:
            return asset_df_test

        # If we filtered this function
        if (not self.accepted):
            asset_df_test[pred_col] = [0 for i in range(asset_df_test.shape[0])]
            return asset_df_test

        # Prepare df_test
        for factor in self.factors:
            asset_df_test.loc[:, (factor)] = (asset_df_test[factor] - self.factor_stat[factor]['mean'])/self.factor_stat[factor]['std']

        y_pred = self.models[model_name].predict(asset_df_test[self.factors])

        # If filter function was specified, then filter the predictions
        if (filter_pred is not None):

            new_y_pred = []

            for yi_pred in list(y_pred):

                if (filter_pred(self, yi_pred, *args, **kwargs)):
                    new_y_pred.append(0)
                else:
                    new_y_pred.append(yi_pred)

            y_pred = new_y_pred

        # Save the predictions in the asset dataframe
        asset_df_test[pred_col] = y_pred

        # Return asset dataframe
        return asset_df_test


class GetData(object):

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
        y   = np.array(df[self.ret_col])
        x   = np.array(df[factors])
        return x, y
    
    '''
    Get divided 80/20 data
    ''' 
    def get_80_20(self, df):
        time = list(set(df.index.get_level_values(0)))
        time.sort()
        train_index = time[                  :int(len(time)*0.8)]
        test_index  = time[int(len(time)*0.8):                  ]
        train = df.loc[train_index]
        test  = df.loc[test_index]
        return train, test