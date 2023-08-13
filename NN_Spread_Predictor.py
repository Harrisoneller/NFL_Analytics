def NFL_NN_SPREAD_PREDICTOR(home_team_input,away_team_input):    
    import Statlete
    import numpy as np 
    import pandas as pd
    import sportsdataverse.nfl as nfl
    import scipy
    import difflib
    import json
    import numpy as np
    #import matplotlib.pyplot as plt
    import time
    import tensorflow as tf 
    import keras
    from sklearn.model_selection import train_test_split
    import elo_scrape as elo_funs
    
    
    class Team:
        import numpy as np 
        import pandas as pd
        import sportsdataverse.nfl as nfl
        import scipy
        import difflib
        import json

        #teams_info = nfl.nfl_loaders.load_nfl_teams()

        def __init__(self, team):
            teams_info = pd.read_csv('teams_info.csv')
            team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
            self.team = team
            self.team_abr = list(teams_info.team_abbr[teams_info.team_name.isin(self.team)])[0]
            #self.game = self.Game(team)
            self.season = self.Season(team)
        
            
            
        class Season:       
            ID = nfl.load_nfl_players()
            stats = nfl.load_nfl_player_stats()
            ngs = nfl.load_nfl_ngs_receiving() 
            
            def __init__(self, team):
                teams_info = pd.read_csv('teams_info.csv')
                team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
                self.team_season = team
                self.team_abr_season = teams_info.team_abbr[teams_info.team_name.isin(team)]
                #self.player_ngs = player_ngs 

            def get_data(self, season = [2021,2022]):
                pbp = nfl.load_nfl_pbp(seasons=(season))
                team_pbp = pbp[(pbp.home_team.isin(self.team_abr_season)) | (pbp.away_team.isin(self.team_abr_season))]
                team_off_pbp = team_pbp[(team_pbp.posteam.isin(self.team_abr_season)) | (team_pbp.defteam.isin(self.team_abr_season)) ]
                
                
                #output1 = player_df[player_df['week'] == week]  
                try:
                    df = team_off_pbp
                except:
                    df = team_off_pbp
                return df
            
    hist_elo = pd.read_csv('nfl_historical_elo.csv')
    hist_elo = hist_elo[hist_elo.season > 2020]

    # ############################################################### data preparation
    TEAM = Team(f'{home_team_input}')
    try:
        dfh = Team.Season(f'{home_team_input}').get_data()
    except:
        try:
            dfh = Team.Season(f'{home_team_input}').get_data()
        except:
            try:
                dfh = Team.Season(f'{home_team_input}').get_data()
            except:
                try:
                    dfh = Team.Season(f'{home_team_input}').get_data()
                except:         
                    try:
                        dfh = Team.Season(f'{home_team_input}').get_data()
                    except:              
                        try:
                            dfh = Team.Season(f'{home_team_input}').get_data()
                        except:             
                            try:
                                dfh = Team.Season(f'{home_team_input}').get_data()
                            except:   
                                pass                                                        
    df_numeric = dfh.groupby(by=['game_id']).mean()
    df_numeric['opp_passing_yards'] = 0
    df_numeric['opp_rushing_yards'] = 0
    df_numeric['OSR'] = 0
    df_numeric['DSR'] = 0
    df_numeric['opp_OSR'] = 0
    df_numeric['opp_DSR'] = 0

    home_team = []
    away_team = []
    total_epa = []
    opp_epa = []
    score_diff = []
    home_field = []
    opp_success = []

    index = 0
    for GID in dfh.game_id.unique():
        home_team.append(dfh[dfh['game_id'] == GID].home_team.iloc[0])
        away_team.append(dfh[dfh['game_id'] == GID].away_team.iloc[0])
        score_diff.append(dfh[dfh['game_id'] == GID].score_differential.iloc[len(dfh[dfh['game_id'] == GID])-1])
        if home_team[index] == TEAM.team_abr:
            total_epa.append(dfh[dfh['game_id'] == GID].total_home_epa.iloc[len(dfh[dfh['game_id'] == GID])-1])
            opp_epa.append(dfh[dfh['game_id'] == GID].total_away_epa.iloc[len(dfh[dfh['game_id'] == GID])-1])
            home_field.append(1)
            
        else:
            total_epa.append(dfh[dfh['game_id'] == GID].total_away_epa.iloc[len(dfh[dfh['game_id'] == GID])-1])
            opp_epa.append(dfh[dfh['game_id'] == GID].total_home_epa.iloc[len(dfh[dfh['game_id'] == GID])-1])
            home_field.append(0)
        
        game_df = dfh[dfh['game_id'].isin([GID])]
        passing_yards = []
        receiving_yards = []
        rushing_yards = []
        opp_passing_yards=[]
        opp_receiving_yards=[]
        opp_rushing_yards=[]

        for play in range(len(game_df)):
            if game_df['posteam'].iloc[play] == TEAM.team_abr:
                passing_yards.append(game_df['passing_yards'].iloc[play]) 
                receiving_yards.append(game_df['receiving_yards'].iloc[play]) 
                rushing_yards.append(game_df['rushing_yards'].iloc[play]) 
                
            else:
                opp_passing_yards.append(game_df['passing_yards'].iloc[play]) 
                opp_receiving_yards.append(game_df['receiving_yards'].iloc[play]) 
                opp_rushing_yards.append(game_df['rushing_yards'].iloc[play]) 
            if play == range(len(game_df))[-1]:
                passing_yards = pd.Series(passing_yards).dropna()
                rushing_yards = pd.Series(rushing_yards).dropna()
                opp_passing_yards = pd.Series(opp_passing_yards).dropna()
                opp_rushing_yards = pd.Series(opp_rushing_yards).dropna()

        df_numeric['passing_yards'].iloc[index] = passing_yards.sum()      
        df_numeric['rushing_yards'].iloc[index] = rushing_yards.sum()
        df_numeric['opp_passing_yards'].iloc[index] = opp_passing_yards.sum()
        df_numeric['opp_rushing_yards'].iloc[index] = opp_rushing_yards.sum()
        
        df_numeric['OSR'].iloc[index] = (game_df[game_df['posteam'] == TEAM.team_abr]['success'].sum())/(game_df[game_df['posteam'] == TEAM.team_abr]['success'].count())
        df_numeric['DSR'].iloc[index] = (game_df[game_df['defteam'] == TEAM.team_abr]['success'].sum())/(game_df[game_df['defteam'] == TEAM.team_abr]['success'].count())
        df_numeric['opp_OSR'].iloc[index] = 1 - df_numeric['OSR'].iloc[index]
        df_numeric['opp_DSR'].iloc[index] = 1 - df_numeric['DSR'].iloc[index]
        
        
        #away_final.append(df[df['game_id'] == GID].away_team.iloc[0])
        index+=1
        print(GID)



        
    for j in range(1): # Set df columns 
        
        df_numeric['home_team'] = home_team
        df_numeric['away_team'] = away_team
        df_numeric['score_diff'] = score_diff
        df_numeric['total_epa'] = total_epa
        df_numeric['opp_epa'] = opp_epa
        df_numeric['home_field'] = home_field


    df_numeric['elo'] = 0
    df_numeric['opp_elo'] = 0
    
    if  Team(f'{home_team_input}').team_abr == 'LV':
        home_elo = hist_elo[(hist_elo.team1 == 'OAK') | (hist_elo.team2 == 'OAK')]
    else:
        home_elo = hist_elo[(hist_elo.team1 == Team(f'{home_team_input}').team_abr ) | (hist_elo.team2 == Team(f'{home_team_input}').team_abr)]

    for i in range(len(df_numeric)):
        try:
            if home_elo['team1'].iloc[i] == Team(f'{home_team_input}').team_abr:
                df_numeric['elo'].iloc[i] = home_elo['elo1_pre'].iloc[i]
                df_numeric['opp_elo'].iloc[i] = home_elo['elo2_pre'].iloc[i]
            else:
                df_numeric['elo'].iloc[i] = home_elo['elo2_pre'].iloc[i]
                df_numeric['opp_elo'].iloc[i] = home_elo['elo1_pre'].iloc[i]
        except:
            continue
        
        
    predictors = ['air_yards','epa','pass_oe','passing_yards','rushing_yards','total_epa','opp_epa','home_field','opp_passing_yards','opp_rushing_yards','OSR','DSR','opp_OSR','opp_DSR','elo','opp_elo']
    ht = df_numeric[['air_yards','score_diff','epa','pass_oe','passing_yards','rushing_yards','total_epa','opp_epa','home_field','opp_passing_yards','opp_rushing_yards','OSR','DSR','opp_OSR','opp_DSR','elo','opp_elo']]




    # ################################################################## data scaling / organizing 



    # demonstrate data normalization with sklearn
    from sklearn.preprocessing import MinMaxScaler
    # create scaler
    scaler = MinMaxScaler()
    # fit scaler on data
    scaler.fit(ht[predictors])
    # apply transform
    normalized = scaler.transform(ht[predictors])
    # inverse transform
    inverse = scaler.inverse_transform(normalized)

    import numpy as np

    #  initial data split
    for i in range(1):
        df_norm = pd.DataFrame(normalized,columns=predictors)
        n = len(df_norm)

        train_df = df_norm[0:int(n*0.8)]
        #test_df = df_norm[int(n*0.7):int(n*0.9)]
        val_df = df_norm[int(n*0.8):]


        X_train_home = train_df[predictors]
        #X_test = test_df[predictors]
        x_val_home = val_df[predictors]

        #xtrain = np.array(X_train)
        #xval = np.array(x_val)
        #xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
        #xtest.reshape(xtest.shape[0],xtest.shape[1],1)

        y_train_home = ht['score_diff'][0:int(n*0.8)]
        #y_test = ht['score_diff'][int(n*0.7):int(n*0.9)]
        y_val_home = ht['score_diff'][int(n*0.8):]

        #ytrain = np.array(y_train)
        #ytest = np.array(y_test)
        #val = np.array(y_val)
        # ytrain.reshape(ytrain.shape[0],ytrain.shape[1],1)
        # ytest.reshape(ytest.shape[0],ytest.shape[1],1)




    # inputs = tf.keras.Input(shape=(len(X_train_home.columns)), name = 'input')
    # hidden1 = tf.keras.layers.Dense(units = 100, activation = 'ReLU', name = 'hidden1')(inputs)
    # hidden2 = tf.keras.layers.Dense(units = 50, activation = 'ReLU', name = 'hidden2')(hidden1)
    # hidden3 = tf.keras.layers.Dense(units = 10, activation = 'ReLU', name = 'hidden3')(hidden2)
    # hidden4 = tf.keras.layers.Dense(units = 5, activation = 'ReLU', name = 'hidden4')(hidden3)
    # hidden5 = tf.keras.layers.Dense(units = 2, activation = 'ReLU', kernel_regularizer = tf.keras.regularizers.L1(.01), name = 'hidden5')(hidden4)
    # output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden5)


    # #hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    # model = 0
    # model = tf.keras.Model(inputs = inputs, outputs = output)
    # model.compile(loss = 'MSE', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))# metrics = ['mae'])
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    # history = model.fit(x = X_train_home,y=y_train_home, validation_data=(x_val_home, y_val_home),callbacks = [stop_early],  batch_size = 1, epochs = 250)







    # ################################################################################## return data back to original format and evaluate performance 


    # df_norm = pd.DataFrame(normalized,columns=predictors)
    # n = len(df_norm)

    # train_df = df_norm[0:int(n*0.7)]
    # test_df = df_norm[int(n*0.7):int(n*0.9)]
    # val_df = df_norm[int(n*0.9):]


    # X_train = train_df[predictors]
    # X_test = test_df[predictors]
    # x_val = val_df[predictors]

    # xtrain = np.array(X_train)
    # xtest = np.array(X_test)

    # y_train = ht['score_diff'][0:int(n*0.7)]
    # y_test = ht['score_diff'][int(n*0.7):int(n*0.9)]
    # y_val = ht['score_diff'][int(n*0.9):]

    # ytrain = np.array(y_train)
    # ytest = np.array(y_test)


    # yhat = model.predict(X_test)
    # model.evaluate(X_test,  y_test, verbose=2)


    # # visualize
    # train_len = np.arange(0,len(y_train))
    # train_len = ht.index[0:len(y_train)]
    # test_len = len(yhat)
    # #actual_nox_len = np.arange(max(train_len),(max(train_len) + test_len))
    # actual_nox_len = ht.index[len(train_len):(len(train_len) + test_len)]
    # val_len = ht.index[(len(train_len) + test_len):]
    # #val_len = test_len[:len(y_val)]


    # plt.scatter(train_len,y_train, color = 'g', label = '(train data)', s = 8)
    # plt.scatter(actual_nox_len,y_test, color = 'blue',label = 'Actual', s=8);#yhat = yhat + 2
    # plt.scatter(actual_nox_len,yhat, color = 'r', label = 'predicted',s=8)
    # plt.scatter(val_len,y_val, color = 'cyan', label = 'val',s=8)
    # plt.legend()
    # plt.show()





























    ###################################################################################################################################################### AWAY ######################################################################################################################################





    # ############################################################### data preparation
    OPP = Team(f'{away_team_input}')
    try:
        dfa = Team.Season(f'{away_team_input}').get_data()
    except:
        try:
            dfa = Team.Season(f'{away_team_input}').get_data()
        except:
            try:
                dfa = Team.Season(f'{away_team_input}').get_data()
            except:
                try:
                    dfa = Team.Season(f'{away_team_input}').get_data()
                except:         
                    try:
                        dfa = Team.Season(f'{away_team_input}').get_data()
                    except:              
                        try:
                            dfa = Team.Season(f'{away_team_input}').get_data()
                        except:             
                            try:
                                dfa = Team.Season(f'{away_team_input}').get_data()
                            except:   
                                pass    
    df2_numeric = dfa.groupby(by=['game_id']).mean()
    df2_numeric['opp_passing_yards'] = 0
    df2_numeric['opp_rushing_yards'] = 0
    df2_numeric['OSR'] = 0
    df2_numeric['DSR'] = 0
    df2_numeric['opp_OSR'] = 0
    df2_numeric['opp_DSR'] = 0

    home_team = []
    away_team = []
    total_epa = []
    opp_epa = []
    score_diff = []
    home_field = []
    opp_success = []

    index = 0

    for GID in dfa.game_id.unique():
        home_team.append(dfa[dfa['game_id'] == GID].home_team.iloc[0])
        away_team.append(dfa[dfa['game_id'] == GID].away_team.iloc[0])
        score_diff.append(dfa[dfa['game_id'] == GID].score_differential.iloc[len(dfa[dfa['game_id'] == GID])-1])
        if away_team[index] == OPP.team_abr:
            total_epa.append(dfa[dfa['game_id'] == GID].total_home_epa.iloc[len(dfa[dfa['game_id'] == GID])-1])
            opp_epa.append(dfa[dfa['game_id'] == GID].total_away_epa.iloc[len(dfa[dfa['game_id'] == GID])-1])
            home_field.append(1)
            
        else:
            total_epa.append(dfa[dfa['game_id'] == GID].total_away_epa.iloc[len(dfa[dfa['game_id'] == GID])-1])
            opp_epa.append(dfa[dfa['game_id'] == GID].total_home_epa.iloc[len(dfa[dfa['game_id'] == GID])-1])
            home_field.append(0)
        
        game_df = dfa[dfa['game_id'].isin([GID])]
        passing_yards = []
        receiving_yards = []
        rushing_yards = []
        opp_passing_yards=[]
        opp_receiving_yards=[]
        opp_rushing_yards=[]

        for play in range(len(game_df)):
            if game_df['posteam'].iloc[play] == OPP.team_abr:
                passing_yards.append(game_df['passing_yards'].iloc[play]) 
                receiving_yards.append(game_df['receiving_yards'].iloc[play]) 
                rushing_yards.append(game_df['rushing_yards'].iloc[play]) 
                
            else:
                opp_passing_yards.append(game_df['passing_yards'].iloc[play]) 
                opp_receiving_yards.append(game_df['receiving_yards'].iloc[play]) 
                opp_rushing_yards.append(game_df['rushing_yards'].iloc[play]) 
            if play == range(len(game_df))[-1]:
                passing_yards = pd.Series(passing_yards).dropna()
                rushing_yards = pd.Series(rushing_yards).dropna()
                opp_passing_yards = pd.Series(opp_passing_yards).dropna()
                opp_rushing_yards = pd.Series(opp_rushing_yards).dropna()

        df2_numeric['passing_yards'].iloc[index] = passing_yards.sum()      
        df2_numeric['rushing_yards'].iloc[index] = rushing_yards.sum()
        df2_numeric['opp_passing_yards'].iloc[index] = opp_passing_yards.sum()
        df2_numeric['opp_rushing_yards'].iloc[index] = opp_rushing_yards.sum()
        
        df2_numeric['OSR'].iloc[index] = (game_df[game_df['posteam'] == OPP.team_abr]['success'].sum())/(game_df[game_df['posteam'] == OPP.team_abr]['success'].count())
        df2_numeric['DSR'].iloc[index] = (game_df[game_df['defteam'] == OPP.team_abr]['success'].sum())/(game_df[game_df['defteam'] == OPP.team_abr]['success'].count())
        df2_numeric['opp_OSR'].iloc[index] = 1 - df2_numeric['OSR'].iloc[index]
        df2_numeric['opp_DSR'].iloc[index] = 1 - df2_numeric['DSR'].iloc[index]
        
        #away_final.append(df[df['game_id'] == GID].away_team.iloc[0])
        index+=1
        print(GID)



        
    for j in range(1): # Set df columns 
        
        df2_numeric['home_team'] = home_team
        df2_numeric['away_team'] = away_team
        df2_numeric['score_diff'] = score_diff
        df2_numeric['total_epa'] = total_epa
        df2_numeric['opp_epa'] = opp_epa
        df2_numeric['home_field'] = home_field


    df2_numeric['elo'] = 0
    df2_numeric['opp_elo'] = 0
    
    if  Team(f'{home_team_input}').team_abr == 'LV':
        away_elo = hist_elo[(hist_elo.team1 == 'OAK' ) | (hist_elo.team2 == 'OAK')]
    else:
        away_elo = hist_elo[(hist_elo.team1 == Team(f'{away_team_input}').team_abr ) | (hist_elo.team2 == Team(f'{away_team_input}').team_abr)]

    for i in range(len(df2_numeric)):
        try:
            if away_elo['team1'].iloc[i] == Team(f'{away_team_input}').team_abr:
                df2_numeric['elo'].iloc[i] = away_elo['elo1_pre'].iloc[i]
                df2_numeric['opp_elo'].iloc[i] = away_elo['elo2_pre'].iloc[i]
            else:
                df2_numeric['elo'].iloc[i] = away_elo['elo2_pre'].iloc[i]
                df2_numeric['opp_elo'].iloc[i] = away_elo['elo1_pre'].iloc[i]
        except:
            continue
        
        
    predictors2 = ['air_yards','epa','pass_oe','passing_yards','rushing_yards','total_epa','opp_epa','home_field','opp_passing_yards','opp_rushing_yards','OSR','DSR','opp_OSR','opp_DSR','elo','opp_elo']
    at = df2_numeric[['air_yards','score_diff','epa','pass_oe','passing_yards','rushing_yards','total_epa','opp_epa','home_field','opp_passing_yards','opp_rushing_yards','OSR','DSR','opp_OSR','opp_DSR','elo','opp_elo']]



    # demonstrate data normalization with sklearn
    from sklearn.preprocessing import MinMaxScaler
    # create scaler
    scaler2 = MinMaxScaler()
    # fit scaler on data
    scaler2.fit(at[predictors])
    # apply transform
    normalized2 = scaler2.transform(at[predictors])
    # inverse transform
    inverse2 = scaler2.inverse_transform(normalized2)

    import numpy as np


    #  initial data split
    for i in range(1):
        df_norm2 = pd.DataFrame(normalized2,columns=predictors2)
        n = len(df_norm2)

        train_df = df_norm2[0:int(n*0.8)]
        #test_df = df_norm[int(n*0.7):int(n*0.9)]
        val_df = df_norm2[int(n*0.8):]


        X_train_away = train_df[predictors2]
        #X_test = test_df[predictors]
        x_val_away = val_df[predictors2]

        #xtrain = np.array(X_train)
        #xval = np.array(x_val)
        #xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
        #xtest.reshape(xtest.shape[0],xtest.shape[1],1)

        y_train_away = at['score_diff'][0:int(n*0.8)]
        #y_test = ht['score_diff'][int(n*0.7):int(n*0.9)]
        y_val_away = at['score_diff'][int(n*0.8):]

        #ytrain = np.array(y_train)
        #ytest = np.array(y_test)
        #yval = np.array(y_val)
        # ytrain.reshape(ytrain.shape[0],ytrain.shape[1],1)
        # ytest.reshape(ytest.shape[0],ytest.shape[1],1)



    # inputs = tf.keras.Input(shape=(len(X_train_away.columns)), name = 'input')
    # hidden1 = tf.keras.layers.Dense(units = 200, activation = 'ReLU', name = 'hidden1')(inputs)
    # hidden2 = tf.keras.layers.Dense(units = 70, activation = 'ReLU', name = 'hidden2')(hidden1)
    # hidden3 = tf.keras.layers.Dense(units = 25, activation = 'ReLU', name = 'hidden3')(hidden2)
    # hidden4 = tf.keras.layers.Dense(units = 10, activation = 'ReLU', name = 'hidden4')(hidden3)
    # hidden5 = tf.keras.layers.Dense(units = 5, activation = 'ReLU', kernel_regularizer = tf.keras.regularizers.L1(.01), name = 'hidden5')(hidden4)
    # output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden5)


    # #hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    # model2 = 0
    # model2 = tf.keras.Model(inputs = inputs, outputs = output)
    # model2.compile(loss = 'MSE', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))# metrics = ['mae'])
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    # history2 = model2.fit(x = X_train_away,y=y_train_away, validation_data=(x_val_away, y_val_away),callbacks = [stop_early],  batch_size = 1, epochs = 250)










    # curr_elo = elo_funs.get_current_elo()
    # ht_elo = curr_elo[curr_elo.index == home_team_input]
    # at_elo = curr_elo[curr_elo.index == away_team_input]


    elo_list = [
        1405,
        1412,
        1543,
        1710,
        1431,
        1336,
        1681,
        1516,
        1615,
        1354,
        1495,
        1557,
        1325,
        1381,
        1517,
        1729,
        1500,
        1523,
        1424,
        1519,
        1569,
        1526,
        1531,
        1450,
        1550,
        1720,
        1533,
        1709,
        1509,
        1501,
        1472,
        1438]

    team_abbrs = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']
    curr_elo = pd.DataFrame({'team':team_abbrs, 'elo':elo_list})
    # ##########                                                                ########################### make predictions 

    ht_elo = float(curr_elo.loc[curr_elo.team == Team(f'{home_team_input}').team_abr,'elo'])
    at_elo = float(curr_elo.loc[curr_elo.team == Team(f'{away_team_input}').team_abr,'elo'])

    # home_pred_array = {'air_yards':ht.air_yards.median(),'epa':ht.epa.median(), 'pass_oe':ht.pass_oe.median(),'passing_yards': ht.passing_yards.median(),'rushing_yards': ht.rushing_yards.median(),'total_epa':ht.total_epa.median(),'opp_epa':at.total_epa.median(),'home_field':1,'opp_passing_yards':at.passing_yards.median(),'opp_rushing_yards':at.rushing_yards.median(),'OSR':ht.OSR.median(),'DSR':ht.DSR.median(),'opp_OSR':at.OSR.median(),'opp_DSR':at.DSR.median(),'elo':ht_elo[0],'opp_elo':at_elo[0]}
    # away_pred_array = {'air_yards':at.air_yards.median(),'epa':at.epa.median(), 'pass_oe':at.pass_oe.median(),'passing_yards': at.passing_yards.median(),'rushing_yards': at.rushing_yards.median(),'total_epa':at.total_epa.median(),'opp_epa':ht.total_epa.median(),'home_field':0,'opp_passing_yards':ht.passing_yards.median(),'opp_rushing_yards':ht.rushing_yards.median(),'OSR':at.OSR.median(),'DSR':at.DSR.median(),'opp_OSR':ht.OSR.median(),'opp_DSR':ht.DSR.median(),'elo':at_elo[0],'opp_elo':ht_elo[0]}


    home_pred_array = {'air_yards':ht.air_yards.median(),'epa':ht.epa.median(), 'pass_oe':ht.pass_oe.median(),'passing_yards': ht.passing_yards.median(),'rushing_yards': ht.rushing_yards.median(),'total_epa':ht.total_epa.median(),'opp_epa':at.total_epa.median(),'home_field':1,'opp_passing_yards':at.passing_yards.median(),'opp_rushing_yards':at.rushing_yards.median(),'OSR':ht.OSR.median(),'DSR':ht.DSR.median(),'opp_OSR':at.OSR.median(),'opp_DSR':at.DSR.median(),'elo':ht_elo,'opp_elo':at_elo}
    away_pred_array = {'air_yards':at.air_yards.median(),'epa':at.epa.median(), 'pass_oe':at.pass_oe.median(),'passing_yards': at.passing_yards.median(),'rushing_yards': at.rushing_yards.median(),'total_epa':at.total_epa.median(),'opp_epa':ht.total_epa.median(),'home_field':0,'opp_passing_yards':ht.passing_yards.median(),'opp_rushing_yards':ht.rushing_yards.median(),'OSR':at.OSR.median(),'DSR':at.DSR.median(),'opp_OSR':ht.OSR.median(),'opp_DSR':ht.DSR.median(),'elo':at_elo,'opp_elo':ht_elo}




    # PH = scaler.transform(home_pred_array.reshape(1, -1) )
    # PA = scaler2.transform(away_pred_array.reshape(1, -1) )
    df_home  = pd.DataFrame(data = home_pred_array, index=np.arange(0,1))
    df_away  = pd.DataFrame(data = away_pred_array, index=np.arange(0,1))
    # pd.DataFrame(away_pred_array)
    PH = scaler.transform(df_home)
    PA = scaler2.transform(df_away)

    yhat_home=[]
    yhat_away=[]
    for sim in range(0,10):
    ################# home ################ 
        inputs = tf.keras.Input(shape=(len(X_train_home.columns)), name = 'input')
        hidden1 = tf.keras.layers.Dense(units = 500, activation = 'ReLU', name = 'hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units = 200, activation = 'ReLU', name = 'hidden2')(hidden1)
        dropout1 = tf.keras.layers.Dropout(rate = .2)(hidden2)
        hidden3 = tf.keras.layers.Dense(units = 50, activation = 'ReLU', name = 'hidden3')(dropout1)
        dropout2 = tf.keras.layers.Dropout(rate = .2)(hidden3)
        hidden4 = tf.keras.layers.Dense(units = 15, activation = 'ReLU', name = 'hidden4')(dropout2)
        dropout3 = tf.keras.layers.Dropout(rate = .2)(hidden4)
        hidden5 = tf.keras.layers.Dense(units = 5, activation = 'ReLU', kernel_regularizer = tf.keras.regularizers.L1(.01), name = 'hidden5')(dropout3)
        output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden5)


        #hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model = 0
        model = tf.keras.Model(inputs = inputs, outputs = output)
        model.compile(loss = 'MAE', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))# metrics = ['mae'])
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=33)

        history = model.fit(x = X_train_home,y=y_train_home, validation_data=(x_val_home, y_val_home),callbacks = [stop_early],  batch_size = 1, epochs = 100)
        
        yhat_home.append(model.predict(PH))
        ############## away ##########
        inputs = tf.keras.Input(shape=(len(X_train_away.columns)), name = 'input')
        hidden1 = tf.keras.layers.Dense(units = 500, activation = 'ReLU', name = 'hidden1')(inputs)
        hidden2 = tf.keras.layers.Dense(units = 200, activation = 'ReLU', name = 'hidden2')(hidden1)
        dropout1 = tf.keras.layers.Dropout(rate = .2)(hidden2)
        hidden3 = tf.keras.layers.Dense(units = 50, activation = 'ReLU', name = 'hidden3')(dropout1)
        dropout2 = tf.keras.layers.Dropout(rate = .2)(hidden3)
        hidden4 = tf.keras.layers.Dense(units = 15, activation = 'ReLU', name = 'hidden4')(dropout2)
        dropout3 = tf.keras.layers.Dropout(rate = .2)(hidden4)
        hidden5 = tf.keras.layers.Dense(units = 5, activation = 'ReLU', kernel_regularizer = tf.keras.regularizers.L1(.01), name = 'hidden5')(dropout3)
        output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden5)



        #hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model2 = 0
        model2 = tf.keras.Model(inputs = inputs, outputs = output)
        model2.compile(loss = 'MAE', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))# metrics = ['mae'])
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=33)

        history2 = model2.fit(x = X_train_away,y=y_train_away, validation_data=(x_val_away, y_val_away),callbacks = [stop_early],  batch_size = 1, epochs = 100)

        yhat_away.append(model2.predict(PA))

    output_data = dict({f'{home_team_input}_Spread':np.mean(yhat_home), f'{away_team_input}_Spread':np.mean(yhat_away)})

    #return print(yhat_home), print(yhat_away)
    Spread_Predictions = pd.DataFrame(data = output_data,index=np.arange(0,1))
    Spread_Predictions['Midpoint'] = round(( (Spread_Predictions[f'{home_team_input}_Spread'] + Spread_Predictions[f'{away_team_input}_Spread'] )/2),1)
    

    return Spread_Predictions



