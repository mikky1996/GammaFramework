import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from AssetsData import AssetsData

class VizualExperiment(object):

    def __init__(self, asset_col = 'ID', time_col = 'TIMESTAMP', ret_col = 'y'):
        self.asset_col  = asset_col
        self.time_col   = time_col
        self.ret_col    = ret_col
        self.assetsdata = AssetsData()

    '''
    This function plots asset returns histogram and normal distribution
    This can help to get the autoregressive nature of the stocks
    '''
    def hist_asset_returns(self, asset_name, df, bins = 100, image_path = None):

        # Get the data
        df_a = self.assetsdata.get_asset_df(asset_name, df)
        y = list(df_a[self.ret_col])
        y.sort()

        # Get data about normal distribution
        hmean = np.mean(y)
        hstd = np.std(y)
        pdf = stats.norm.pdf(y, hmean, hstd)

        # Plot all together
        plt.title("Mean {} and std {} factor is {}".format(hmean, hstd, asset_name))
        plt.plot(y, pdf)
        plt.hist(y, bins = bins)
        
        # Save the picture if needed
        if (image_path is not None):
            plt.savefig(image_path)

        plt.show()

    '''
    This function plots returns and predictions of regressor
    '''
    def plot_prediction(self, asset_name, reg, df, factors = None, image_path = None):
        
        # Get asset specific dataframe
        asset_df = self.assetsdata.get_asset_df_without_nan(asset_name, df)
        x, y     = self.assetsdata.get_x_y(asset_df, factors = factors)

        # Fit the data
        reg.fit(x,y)
        y_pred   = reg.predict(x)
        plt.title("'Asset name {}, with R^2 = {}'\n{}".format(asset_name, reg.score(x,y), reg))
        plt.plot(range(len(y_pred)), y_pred, label = 'y_pred')
        plt.plot(range(len(y))     ,      y, label = 'y_real')
        plt.legend()

        # Save the picture if needed
        if (image_path is not None):
            plt.savefig(image_path)

        plt.show()
        return reg.coef_