import sportsdataverse.nfl as nfl 
from bs4 import BeautifulSoup
#import urllib.request
import requests
import pandas as pd
import numpy as np
import scipy
import difflib
import json
#import matplotlib.pyplot as plt
import time
import scipy
import statistics as stats
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#import xgboost as xgb
from sklearn.model_selection import train_test_split
# Calculate ELO rating for NFL teams using data scraped from web
# Starting out with calculating # of wins
from nfl_dev import nfl_dev
import tensorflow as tf
import nfl_data_py as nfl_dp

class nfl_model:
    
    def __init__(self):
        from nfl_dev import nfl_dev 
        new_index = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
       'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LA', 'MIA', 'MIN',
       'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN',
       'WAS']

        self.season=2024
        
        out_elo, self.team_ref_2022= nfl_dev().get_current_elo(input_season=2022)
        self.team_ref_2022['season']=None
        self.team_ref_2022['season']=2022
        
        out_elo, self.team_ref_2023= nfl_dev().get_current_elo(input_season=2023)
        self.team_ref_2023['season']=None
        self.team_ref_2023['season']=2023
        
        out_elo, self.team_ref_2024= nfl_dev().get_current_elo(input_season=2024)
        self.team_ref_2024['season']=None
        self.team_ref_2024['season']=2024
        
        self.team_ref_2022.index = new_index
        self.team_ref_2023.index = new_index
        self.team_ref_2024.index = new_index


    def model(self):
        new_index = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
            'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LA', 'MIA', 'MIN',
            'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN',
            'WAS']

        out_elo, team_ref_2022= nfl_dev().get_current_elo(input_season=2022)
        team_ref_2022['season']=None
        team_ref_2022['season']=2022
        out_elo, team_ref_2023= nfl_dev().get_current_elo(input_season=2023)
        team_ref_2023['season']=None
        team_ref_2023['season']=2023
        out_elo, team_ref_2024= nfl_dev().get_current_elo(input_season=2024)
        team_ref_2024['season']=None
        team_ref_2024['season']=2024
        team_ref_2022.index = new_index
        team_ref_2023.index = new_index
        team_ref_2024.index = new_index

        len(team_ref_2022.loc['ATL','elo'])

        ######################################################################################################################################################################################
        import nfl_data_py as nfl_data
        week = 2

        s = nfl_data.import_schedules([2024])
        s = s.loc[s['week'] == week,]
        gid = [] 
        for game in range(len(s)):
            gid.append(s['game_id'].iloc[game])

        df = pd.DataFrame(columns = ['Home_Team', 'Home_Score','Away_Team', 'Away_Score','Spread','Total'], index = gid )

        teams = nfl_data.import_team_desc()




        pbp = nfl.load_nfl_pbp(seasons=([2022, 2023,2024]),return_as_pandas=True)


        # pbp = pbp[(pbp.kickoff_attempt == 0) & (pbp.punt_attempt == 0) & (pbp.field_goal_attempt == 0)
        #           & (pbp.down.isin([1,2,3,4])) & (pbp.penalty == 0)]
        pbp = pbp[((pbp['pass'] == 1) | (pbp['rush'] == 1)) & ( pbp.season_type == "REG")]
        mask = {'home':1,'away':0}
        pbp.posteam_type = pbp.posteam_type.map(mask)

        pbp['elo']=1400

        for play in pbp.index:
            if pbp.loc[play,'season'] == 2022:
                pbp.loc[play,'elo'] = team_ref_2022.loc[pbp.loc[play,'posteam'],'elo'][int(pbp.loc[play,'week'])-1]
            elif pbp.loc[play,'season'] == 2023:
                pbp.loc[play,'elo'] = team_ref_2023.loc[pbp.loc[play,'posteam'],'elo'][int(pbp.loc[play,'week'])-1]
            else:
                pbp.loc[play,'elo'] = team_ref_2024.loc[pbp.loc[play,'posteam'],'elo'][int(pbp.loc[play,'week'])-1]

                

        cols = pbp.select_dtypes(include = 'number').columns
        df_off = pbp.groupby(['posteam','game_id'])[cols].agg(['min','max','sum','median','size'])
        df_def = pbp.groupby(['defteam','game_id'])[cols].agg(['min','max','sum','median','size'])

        test = pbp.groupby(['posteam','defteam','game_id'])[cols].agg(['min','max','sum','median','size'])

        df_def = df_def.add_suffix('_allowed')
        df_off.index = df_off.index.rename(['team','game_id']);df_def.index = df_def.index.rename(['team','game_id'])


        df = df_off.join(df_def)

        # data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
        # data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']
        # columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )




        #df_off.loc[('DAL', ...)]



        data = pd.DataFrame({'epa':None, 'pass_ypa':None, 'rush_ypa':None, 'turnovers':None, 'takeaways':None,'home_away':None, 'elo':None,
                    'pressures_allowed':None, 'total_plays':None, 
                    
                    'epa_allowed':None, 'pass_ypa_allowed':None, 'rush_ypa_allowed':None, 'opponent_elo':None,
                    'pressures':None, 'total_plays_allowed':None, }, index = [])


        for team in df.index.levels[0]:
            
            temp = pd.DataFrame({'team':team,
                        'epa':df.loc[(team, ...)]['epa']['median'], 
                        'pass_ypa':df.loc[(team, ...)]['passing_yards']['sum']/(df.loc[(team, ...)]['pass_attempt']['sum'] - df.loc[(team, ...)]['sack']['sum']), 
                        'rush_ypa':df.loc[(team, ...)]['rushing_yards']['sum']/(df.loc[(team, ...)]['rush_attempt']['sum']),
                        'turnovers':df.loc[(team, ...)]['interception']['sum']+df.loc[(team, ...)]['fumble']['sum'], 
                        'takeaways':df.loc[(team, ...)]['interception_allowed']['sum_allowed']+df.loc[(team, ...)]['fumble_allowed']['sum_allowed'],
                        'home_away':df.loc[(team, ...)]['posteam_type']['median'],
                        'elo':df.loc[(team, ...)]['elo']['median'],
                        'points': df.loc[(team, ...)]['posteam_score']['max'],
                        'pressures_allowed':df.loc[(team, ...)]['sack']['sum'], 
                        'total_plays':df.loc[(team, ...)]['play_id']['size'], 
                        'epa_allowed':df.loc[(team, ...)]['epa_allowed']['median_allowed'], 
                        'pass_ypa_allowed':df.loc[(team, ...)]['passing_yards_allowed']['sum_allowed']/(df.loc[(team, ...)]['pass_attempt_allowed']['sum_allowed'] - df.loc[(team, ...)]['sack_allowed']['sum_allowed']), 
                        'rush_ypa_allowed':df.loc[(team, ...)]['rushing_yards_allowed']['sum_allowed']/(df.loc[(team, ...)]['rush_attempt_allowed']['sum_allowed']), 
                        'opponent_elo':df.loc[(team, ...)]['elo_allowed']['median_allowed'],
                        'pressures':df.loc[(team, ...)]['sack_allowed']['sum_allowed'], 
                        'total_plays_allowed':df.loc[(team, ...)]['play_id_allowed']['size_allowed']},
                        index = df.loc[(team, ...)].index )
            data = pd.concat([data,temp])




        from sklearn.model_selection import train_test_split

        for col in data.columns:
            if col not in ['team']:
                data[col] = data[col].astype('float32')
                
        data.opponent_elo = data.opponent_elo.astype('float32')
        data.elo = data.elo.astype('float32')
        data.home_away = data.home_away.astype('int')

        train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

        X_train = train_df.drop(['points','team'],axis=1)
        y_train = train_df.points
        X_test = test_df.drop(['points','team'],axis=1)
        y_test = test_df.points




        inputs = tf.keras.Input(shape=(len(X_train.columns),), name = 'input')
        hidden1 = tf.keras.layers.Dense(units = 1000, activation = 'relu', name = 'hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units = 500, activation = 'relu', name = 'hidden2')(hidden1)
        dropout1 = tf.keras.layers.Dropout(rate = .2)(hidden2)
        hidden3 = tf.keras.layers.Dense(units = 200, activation = 'relu', name = 'hidden3')(dropout1)
        dropout2 = tf.keras.layers.Dropout(rate = .2)(hidden3)
        hidden4 = tf.keras.layers.Dense(units = 50, activation = 'relu', name = 'hidden4')(dropout2)
        dropout3 = tf.keras.layers.Dropout(rate = .2)(hidden4)
        hidden5 = tf.keras.layers.Dense(units = 10, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L1(.01), name = 'hidden5')(dropout3)
        output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden5)


        #hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model = 0
        model = tf.keras.Model(inputs = inputs, outputs = output)
        model.compile(loss = 'mae', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))# metrics = ['mae'])
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)

        history = model.fit(x = X_train,y=y_train, validation_split=.15,callbacks = [stop_early],  batch_size = 2, epochs = 200)
    
        yhat = model.predict(X_test)
        print( np.mean( yhat - y_test.values))
        
        return model, df


    def XGB(self):
        
        new_index = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
            'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LA', 'MIA', 'MIN',
            'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN',
            'WAS']

        out_elo, team_ref_2022= nfl_dev().get_current_elo(input_season=2022)
        team_ref_2022['season']=None
        team_ref_2022['season']=2022
        out_elo, team_ref_2023= nfl_dev().get_current_elo(input_season=2023)
        team_ref_2023['season']=None
        team_ref_2023['season']=2023
        out_elo, team_ref_2024= nfl_dev().get_current_elo(input_season=2024)
        team_ref_2024['season']=None
        team_ref_2024['season']=2024
        team_ref_2022.index = new_index
        team_ref_2023.index = new_index
        team_ref_2024.index = new_index

        len(team_ref_2022.loc['ATL','elo'])

        ######################################################################################################################################################################################
        import nfl_data_py as nfl_data
        week = self.week

        s = nfl_data.import_schedules([2024])
        s = s.loc[s['week'] == week,]
        gid = [] 
        for game in range(len(s)):
            gid.append(s['game_id'].iloc[game])

        df = pd.DataFrame(columns = ['Home_Team', 'Home_Score','Away_Team', 'Away_Score','Spread','Total'], index = gid )

        teams = nfl_data.import_team_desc()




        pbp = nfl.load_nfl_pbp(seasons=([2022, 2023,2024]),return_as_pandas=True)


        # pbp = pbp[(pbp.kickoff_attempt == 0) & (pbp.punt_attempt == 0) & (pbp.field_goal_attempt == 0)
        #           & (pbp.down.isin([1,2,3,4])) & (pbp.penalty == 0)]
        pbp = pbp[((pbp['pass'] == 1) | (pbp['rush'] == 1)) & ( pbp.season_type == "REG")]
        mask = {'home':1,'away':0}
        pbp.posteam_type = pbp.posteam_type.map(mask)

        pbp['elo']=1400

        for play in pbp.index:
            if pbp.loc[play,'season'] == 2022:
                pbp.loc[play,'elo'] = team_ref_2022.loc[pbp.loc[play,'posteam'],'elo'][int(pbp.loc[play,'week'])-1]
            elif pbp.loc[play,'season'] == 2023:
                pbp.loc[play,'elo'] = team_ref_2023.loc[pbp.loc[play,'posteam'],'elo'][int(pbp.loc[play,'week'])-1]
            else:
                pbp.loc[play,'elo'] = team_ref_2024.loc[pbp.loc[play,'posteam'],'elo'][int(pbp.loc[play,'week'])-1]

                

        cols = pbp.select_dtypes(include = 'number').columns
        df_off = pbp.groupby(['posteam','game_id'])[cols].agg(['min','max','sum','median','size'])
        df_def = pbp.groupby(['defteam','game_id'])[cols].agg(['min','max','sum','median','size'])

        test = pbp.groupby(['posteam','defteam','game_id'])[cols].agg(['min','max','sum','median','size'])

        df_def = df_def.add_suffix('_allowed')
        df_off.index = df_off.index.rename(['team','game_id']);df_def.index = df_def.index.rename(['team','game_id'])


        df = df_off.join(df_def)

        # data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
        # data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']
        # columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )




        #df_off.loc[('DAL', ...)]



        data = pd.DataFrame({'epa':None, 'pass_ypa':None, 'rush_ypa':None, 'turnovers':None, 'takeaways':None,'home_away':None, 'elo':None,
                    'pressures_allowed':None, 'total_plays':None, 
                    
                    'epa_allowed':None, 'pass_ypa_allowed':None, 'rush_ypa_allowed':None, 'opponent_elo':None,
                    'pressures':None, 'total_plays_allowed':None, }, index = [])


        for team in df.index.levels[0]:
            
            temp = pd.DataFrame({'team':team,
                        'epa':df.loc[(team, ...)]['epa']['median'], 
                        'pass_ypa':df.loc[(team, ...)]['passing_yards']['sum']/(df.loc[(team, ...)]['pass_attempt']['sum'] - df.loc[(team, ...)]['sack']['sum']), 
                        'rush_ypa':df.loc[(team, ...)]['rushing_yards']['sum']/(df.loc[(team, ...)]['rush_attempt']['sum']),
                        'turnovers':df.loc[(team, ...)]['interception']['sum']+df.loc[(team, ...)]['fumble']['sum'], 
                        'ypp':(df.loc[(team, ...)]['yards_gained']['sum']/df.loc[(team, ...)]['yards_gained']['size']),
                        'takeaways':df.loc[(team, ...)]['interception_allowed']['sum_allowed']+df.loc[(team, ...)]['fumble_allowed']['sum_allowed'],
                        'home_away':df.loc[(team, ...)]['posteam_type']['median'],
                        'elo':df.loc[(team, ...)]['elo']['median'],
                        'points': df.loc[(team, ...)]['posteam_score']['max'],
                        'pressures_allowed':df.loc[(team, ...)]['sack']['sum'], 
                        'total_plays':df.loc[(team, ...)]['play_id']['size'], 
                        'epa_allowed':df.loc[(team, ...)]['epa_allowed']['median_allowed'], 
                        'pass_ypa_allowed':df.loc[(team, ...)]['passing_yards_allowed']['sum_allowed']/(df.loc[(team, ...)]['pass_attempt_allowed']['sum_allowed'] - df.loc[(team, ...)]['sack_allowed']['sum_allowed']), 
                        'rush_ypa_allowed':df.loc[(team, ...)]['rushing_yards_allowed']['sum_allowed']/(df.loc[(team, ...)]['rush_attempt_allowed']['sum_allowed']), 
                        'ypp_allowed':(df.loc[(team, ...)]['yards_gained_allowed']['sum_allowed']/df.loc[(team, ...)]['yards_gained_allowed']['size_allowed']),
                        'opponent_elo':df.loc[(team, ...)]['elo_allowed']['median_allowed'],
                        'pressures':df.loc[(team, ...)]['sack_allowed']['sum_allowed'], 
                        'total_plays_allowed':df.loc[(team, ...)]['play_id_allowed']['size_allowed']},
                        index = df.loc[(team, ...)].index )
            data = pd.concat([data,temp])




        from sklearn.model_selection import train_test_split
        
# First XGBoost model for Pima Indians dataset
        from numpy import loadtxt
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # load data

        for col in data.columns:
            if col not in ['team']:
                data[col] = data[col].astype('float32')
                
        data.opponent_elo = data.opponent_elo.astype('float32')
        data.elo = data.elo.astype('float32')
        data.home_away = data.home_away.astype('int')

        train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

        X_train = train_df.drop(['points','team'],axis=1)
        y_train = train_df.points
        X_test = test_df.drop(['points','team'],axis=1)
        y_test = test_df.points
        
                # split data into X and y

        model = XGBRegressor()
        model.fit(X_train, y_train)
        # make predictions for test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        
        return model,df
        

        # evaluate predictions

    def projections(self,week,ypp_input=None):
        import nfl_data_py as nfl_dp
        self.week=week
        model,df = self.XGB()
        
        
        s = nfl_dp.import_schedules(years=[2024])
        teams=nfl_dp.import_team_desc()
        s = s[s.week == week]
        projections = pd.DataFrame({'home':None,'home_score':None,'away_score':None,'away':None},index = s.game_id)

        df = df[df[('season','median')] == 2024]

        
        for game in s.index:
            
            home_abbr = s.loc[game, 'home_team']
            away_abbr = s.loc[game, 'away_team']
            home_team_input = f"{teams.loc[teams['team_abbr'] == home_abbr,'team_name'].values[0]}"
            away_team_input = f"{teams.loc[teams['team_abbr'] == away_abbr,'team_name'].values[0]}"
            
            try: 
                
                print('custom ypp input received')
                home_ypp = ypp_input.loc[ypp_input['home'] == home_abbr,'home_off_ypa'],
                away_ypp = ypp_input.loc[ypp_input['away'] == away_abbr,'away_off_ypa']
                home_ypp_allowed = ypp_input.loc[ypp_input['home'] == home_abbr,'home_def_ypa'],
                away_ypp_allowed = ypp_input.loc[ypp_input['away'] == away_abbr,'away_def_ypa']
            except:
                home_ypp = (np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['yards_gained']['sum']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['yards_gained']['size'])) +
                            np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['yards_gained_allowed']['sum_allowed']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['yards_gained_allowed']['size_allowed'])))/2,
                
                away_ypp = (np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['yards_gained']['sum']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['yards_gained']['size'])) +
                            np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['yards_gained_allowed']['sum_allowed']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['yards_gained_allowed']['size_allowed'])))/2,
            


            exp_home = pd.DataFrame({
            'epa':(np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['epa']['median']) + np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['epa_allowed']['median_allowed']))/2,
            'pass_ypa': (np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['passing_yards']['sum']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['pass_attempt']['sum'] - df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['sack']['sum']), ) +
                         np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['passing_yards_allowed']['sum_allowed']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['pass_attempt_allowed']['sum_allowed'] - df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['sack_allowed']['sum_allowed']), ))/2,
            
            'rush_ypa':(np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rushing_yards']['sum']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rush_attempt']['sum'])) +
                        np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rushing_yards_allowed']['sum_allowed']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rush_attempt_allowed']['sum_allowed'])))/2,
            
            
            'turnovers':(np.mean(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['interception']['sum']+df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['fumble']['sum'])+
                         np.mean(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['interception_allowed']['sum_allowed']+df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['fumble_allowed']['sum_allowed']))/2, 
            
            'takeaways':(np.mean(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['interception_allowed']['sum_allowed']+df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['fumble_allowed']['sum_allowed']) +
                         np.mean(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['interception']['sum']+df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['fumble']['sum']))/2,
            'home_away':1,
            'elo':self.team_ref_2024.loc[home_abbr,'Current ELO'],
            'pressures_allowed':(np.median(df.loc[(home_abbr, ...)]['sack']['sum']) + np.median(df.loc[(away_abbr, ...)]['sack_allowed']['sum_allowed']))/2, 
            
            'total_plays':(np.median(df.loc[(home_abbr, ...)]['play_id']['size']) + np.median(df.loc[(away_abbr, ...)]['play_id_allowed']['size_allowed']) )/2, 
            
            'epa_allowed':(np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['epa_allowed']['median_allowed']) + np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['epa']['median']))/2,
            
            'pass_ypa_allowed': (np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['passing_yards_allowed']['sum_allowed']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['pass_attempt_allowed']['sum_allowed'] - df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['sack_allowed']['sum_allowed']), ) +
                                 np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['passing_yards']['sum']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['pass_attempt']['sum'] - df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['sack']['sum']), ))/2,
            
            'rush_ypa_allowed':(np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rushing_yards_allowed']['sum_allowed']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rush_attempt_allowed']['sum_allowed'])) +
                                np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rushing_yards']['sum']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rush_attempt']['sum'])))/2,
            
            
            'opponent_elo':self.team_ref_2024.loc[away_abbr,'Current ELO'],
            'pressures':(np.median(df.loc[(away_abbr, ...)]['sack']['sum']) + np.median(df.loc[(home_abbr, ...)]['sack_allowed']['sum_allowed']))/2, 
            
            'total_plays_allowed':(np.median(df.loc[(away_abbr, ...)]['play_id']['size']) + np.median(df.loc[(home_abbr, ...)]['play_id_allowed']['size_allowed']) )/2,
            'ypp':home_ypp,
            'ypp_allowed':away_ypp,
            
            }, 
                                
            index = [home_abbr] )
            
            
            exp_away = pd.DataFrame({
            'epa':(np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['epa']['median']) + np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['epa_allowed']['median_allowed']))/2,
            'pass_ypa': (np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['passing_yards']['sum']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['pass_attempt']['sum'] - df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['sack']['sum']), ) +
                         np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['passing_yards_allowed']['sum_allowed']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['pass_attempt_allowed']['sum_allowed'] - df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['sack_allowed']['sum_allowed']), ))/2,
            
            'rush_ypa':(np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rushing_yards']['sum']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rush_attempt']['sum'])) +
                        np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rushing_yards_allowed']['sum_allowed']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rush_attempt_allowed']['sum_allowed'])))/2,
                        
            'turnovers':(np.mean(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['interception']['sum']+df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['fumble']['sum'])+
                         np.mean(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['interception_allowed']['sum_allowed']+df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['fumble_allowed']['sum_allowed']))/2, 
            

            
            'takeaways':(np.mean(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['interception_allowed']['sum_allowed']+df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['fumble_allowed']['sum_allowed']) +
                         np.mean(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['interception']['sum']+df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['fumble']['sum']))/2,
            'home_away':0,
            'elo':self.team_ref_2024.loc[away_abbr,'Current ELO'],
            'pressures_allowed':(np.median(df.loc[(away_abbr, ...)]['sack']['sum']) + np.median(df.loc[(home_abbr, ...)]['sack_allowed']['sum_allowed']))/2, 
            
            'total_plays':(np.median(df.loc[(away_abbr, ...)]['play_id']['size']) + np.median(df.loc[(home_abbr, ...)]['play_id_allowed']['size_allowed']) )/2, 
            
            'epa_allowed':(np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['epa_allowed']['median_allowed']) + np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['epa']['median']))/2,
            
            'pass_ypa_allowed': (np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['passing_yards_allowed']['sum_allowed']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['pass_attempt_allowed']['sum_allowed'] - df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['sack_allowed']['sum_allowed']), ) +
                                 np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['passing_yards']['sum']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['pass_attempt']['sum'] - df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['sack']['sum']), ))/2,
            
            'rush_ypa_allowed':(np.median(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rushing_yards_allowed']['sum_allowed']/(df.loc[(away_abbr, ...)][(len(df.loc[(away_abbr, ...)])-5):]['rush_attempt_allowed']['sum_allowed'])) +
                                np.median(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rushing_yards']['sum']/(df.loc[(home_abbr, ...)][(len(df.loc[(home_abbr, ...)])-5):]['rush_attempt']['sum'])))/2,
            
           
            'opponent_elo':self.team_ref_2024.loc[home_abbr,'Current ELO'],
            'pressures':(np.median(df.loc[(home_abbr, ...)]['sack']['sum']) + np.median(df.loc[(away_abbr, ...)]['sack_allowed']['sum_allowed']))/2, 
            
            'total_plays_allowed':(np.median(df.loc[(home_abbr, ...)]['play_id']['size']) + np.median(df.loc[(away_abbr, ...)]['play_id_allowed']['size_allowed']) )/2 ,
            
            'ypp':away_ypp,
            'ypp_allowed':home_ypp,
            }, 
                                
            index = [away_abbr] )
            
            home_score = model.predict(exp_home)
            away_score = model.predict(exp_away)
            
            # projections.loc[s.loc[game,'game_id'],'home_score'] = home_score[0][0]
            # projections.loc[s.loc[game,'game_id'],'away_score'] = away_score[0][0]
            projections.loc[s.loc[game,'game_id'],'home_score'] = home_score[0]
            projections.loc[s.loc[game,'game_id'],'away_score'] = away_score[0]
            projections.loc[s.loc[game,'game_id'],'home'] = home_abbr
            projections.loc[s.loc[game,'game_id'],'away'] = away_abbr
            
            
            
            
        return projections

            

#exp = nfl_model().projections(week=2,ypp_input=None)
