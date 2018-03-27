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
        df_without_nan = df[col].fillna(df[col].mean())
        y   = list(df['y'])
        #x   = list(np.array(df_without_nan))
        return y, df_without_nan 

    
class Assets:
    
    def __init__(self, name, factors = None, model = None, mean = None, var = None, score = None, day_count = None):
        self.name = name
        self.factors = factors
        self.model = model
        self.mean = mean
        self.var = var
        self.score = score
        self.day_count = day_count
        
def sharpe_turnover(df, timecol='TIMESTAMP', wgtcol='y_pred', retcol='y', turnover_penalty=0.1):
        dlywgts = df.groupby(timecol)[wgtcol].apply(
                lambda x: x.abs().sum()).reset_index().rename(columns={wgtcol:'dailywgt'})

        df = df.merge(dlywgts)
        df['wgt'] = df[wgtcol] / df['dailywgt']

        df['wgtret'] = df['wgt'] * df[retcol]
        dlyret = df.groupby(timecol)['wgtret'].sum()
        sharpe = np.sqrt(252) * dlyret.mean() / dlyret.std()

        pt = df.pivot_table('wgt',index='TIMESTAMP',columns='ID')
        turnover = (pt - pt.shift(1)).abs().sum(axis=1).mean()

        print "SR:",sharpe,"TO:",turnover,"Score:",sharpe-turnover_penalty*turnover
        if sharpe > 100.0:
            print 'You are probably overfit' 

        return sharpe - turnover_penalty * turnover
        
        
def delete_half(y_current, scores, ids):
    total_num_assets = len(ids)
    for i in range(total_num_assets):
        id_current = ids[i]
        if int(id_current[1:]) < len(scores):
            if scores[int(id_current[1:])] < 0.03:
                y_current[i] = 0
        if(sum([1 for i in y_current if i == 0]) > total_num_assets/2 - 2):
            break       
        
        
    