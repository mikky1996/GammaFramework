import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AssetsData import AssetsData
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GridSearchCV

class Evaluation(object):

    def __init__(self, asset_col = 'ID', time_col = 'TIMESTAMP', ret_col = 'y', pred_col = 'y_pred'):
        self.asset_col  = asset_col
        self.time_col   = time_col
        self.ret_col    = ret_col
        self.pred_col   = pred_col
        self.assetsdata = AssetsData()

    '''
    This function computes dayly potfolio returns
    '''
    def portfolio_ret(self, df, pred_col = None):
        if (pred_col is not None):
            return df.groupby(df.index.get_level_values(0)).apply(lambda df: (df[self.ret_col]*df[pred_col]).sum())
        else:
            return df.groupby(df.index.get_level_values(0)).apply(lambda df: (df[self.ret_col]*df[self.pred_col]).sum())

    '''
    IC_spearmanr takes as input DF with first indexing by time and second as assets
    IC_spearmanr returns DF with dayli spearman correlation between predictions and real returns
    '''
    def IC_spearmanr(self, df, pred_col = None):
        if (pred_col is not None):
            return df.groupby(df.index.get_level_values(0)).apply(lambda df: spearmanr(df[self.ret_col], df[pred_col])[0])
        else:
            return df.groupby(df.index.get_level_values(0)).apply(lambda df: spearmanr(df[self.ret_col], df[self.pred_col])[0])
    
    '''
    IC_pearsonr takes as input DF with first indexing by time and second as assets
    IC_pearsonr returns DF with dayli pearsonr correlation between predictions and real returns
    '''
    def IC_pearsonr(self, df):
        if (pred_col is not None):
            return df.groupby(df.index.get_level_values(0)).apply(lambda df: pearsonr(df[self.ret_col], df[pred_col])[0])
        else:
            return df.groupby(df.index.get_level_values(0)).apply(lambda df: pearsonr(df[self.ret_col], df[self.pred_col])[0])

    '''
    Get sharp ratio
    '''
    def SR(self, df, pred_col = None):
    	r = self.portfolio_ret(df, pred_col)
    	return r.mean()/r.std()

    '''
    Get the correlation among the factors and the scatter plot
    '''
    def corr(self, df, col1, col2):
        print("Pearson correlation among factors is {} with p-val {}".format(pearsonr(df[col1], df[col2])[0], pearsonr(df[col1], df[col2])[1]))
        plt.scatter(df[col1], df[col2])
        plt.show()

    '''
    Get Sharp Ratio and plot returns
    '''
    def evaluate(self, df, pred_col = None):
        df1 = df.dropna()
        # If we specified pred_col then we evaluate that column
        if (pred_col is not None):
            df1.loc[:, pred_col] = df1.groupby(df1.index.get_level_values(0))[pred_col].apply(lambda w: w/w.abs().sum())
            r = self.portfolio_ret(df1, pred_col)
            print("SR = {}, with mean {} and std {}".format(self.SR(df1, pred_col), r.mean(), r.std()))
        else:
        # If nothing was specified then we use the defaul pred_col from constructor
            df1.loc[:, self.pred_col] = df1.groupby(df1.index.get_level_values(0))[self.pred_col].apply(lambda w: w/w.abs().sum())
            r = self.portfolio_ret(df1, self.pred_col)
            print("SR = {}, with mean {} and std {}".format(self.SR(df1, self.pred_col), r.mean(), r.std()))
        r.plot()
        plt.title('Returns per day')
        plt.ylabel('Returns per day')
        plt.plot()

    '''
    Find optimal parameters for model for the stock with 'folds' fold cross validation
    model can be for example linear_model.Lasso()
    '''
    def parameter_estimation(self, df, model, min_val, max_val, param_partitions, param_name, asset_name, folds = 5, ret_col = 'y', factors = None):

        results_list = []
        asset_df     = self.assetsdata.get_asset_df(asset_name, df)
        X, y         = self.assetsdata.get_x_y(df, factors = factors)

        step      = (float(max_val - min_val))/param_partitions
        param_val = [(min_val + step * i) for i in range(param_partitions)]
        params    = {param_name : param_val}

        clf = GridSearchCV(model, params, cv = folds)
        res = clf.fit(X, y)
        
        return res.best_params_
                
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
