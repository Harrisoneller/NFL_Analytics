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
# class Team:
#     import numpy as np 
#     import pandas as pd
#     import sportsdataverse.nfl as nfl
#     import scipy
#     import difflib
#     import json

#     #teams_info = nfl.nfl_loaders.load_nfl_teams()

#     def __init__(self,team):
#         teams_info = pd.read_csv('teams_info.csv')
#         team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
#         self.team = team
#         self.team_abr = list(teams_info.team_abbr[teams_info.team_name.isin(self.team)])[0]
#         #self.game = self.Game(team)
#         self.season = self.Season(team)
        
        
        
#     class Season:       
#         ID = nfl.load_nfl_players(return_as_pandas=True)
#         stats = nfl.load_nfl_player_stats(return_as_pandas=True)
#         ngs = nfl.load_nfl_ngs_receiving(return_as_pandas=True) 
        
#         def __init__(self, team):
#             teams_info = pd.read_csv('teams_info.csv')
#             team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
#             self.team_season = team
#             self.team_abr_season = teams_info.team_abbr[teams_info.team_name.isin(team)]
#             #self.player_ngs = player_ngs 

#         def get_data(self, season = [2023,2024],data = False):
#             if data is not False:
#                 print("data received")
#                 pbp = data
#             else:
#                 pbp = nfl.load_nfl_pbp(seasons=(season),return_as_pandas=True)
#             team_pbp = pbp[(pbp.home_team.isin(self.team_abr_season)) | (pbp.away_team.isin(self.team_abr_season))]
#             team_off_pbp = team_pbp[(team_pbp.posteam.isin(self.team_abr_season)) | (team_pbp.defteam.isin(self.team_abr_season)) ]
            
            
#             #output1 = player_df[player_df['week'] == week]  
#             try:
#                 df = team_off_pbp
#             except:
#                 df = team_off_pbp
#             return df
  
    

# def get_current_elo(input_season):
    
#     # Data source we are going to scrape for results
#     data_url = f'https://www.pro-football-reference.com/years/{input_season}/games.htm#games::none'
            

#     # Scrape the index of a given page
#     # Return a list of specified web elements
#     def scrape(selection,parent_object,element_type):

        
#         # Select the given div
#     #  data = soup.findAll("div", { "class" : "table_outer_container" })
        
#         list_links = []
#         data = soup.findAll(parent_object, { "data-stat" : selection })
#         for element in data:
#             #print(element['href'])
            
#             # Add NA option 
#             if element_type!='na':          
#                 list_links += [a.contents[0] for a in element.findAll(element_type)]
#             else:
#                 # Extracts number if it exists
#                 if str(element.renderContents()) != "b''":
#                     list_links += [str(element.renderContents()).split('\'')[1]]
                    
#             #print(list_links)
#     # 
#         return list_links



#     # This is the web data we 
#     page = requests.get(data_url)
#     soup = BeautifulSoup(page.content, 'html.parser')


#     # Automatically only goes as far as shortest list
#     # which is the pts_win (limits to only current games played)
#     import numpy as np
#     # this is a game level dataframe

#     length = len(scrape("pts_win",'td','strong'))

#     week = scrape("week_num",'th','na')

#     # Remove all the text from our week data column
#     while 'Week' in week: week.remove('Week')

#     season = pd.DataFrame(np.column_stack([week[:length],scrape("winner",'td','a')[:length],scrape("loser",'td','a')[:length],scrape("pts_win",'td','strong')[:length],scrape("pts_lose",'td','na')[:length]]),columns=['week','winner','loser',"pts_win",'pts_lose'])

#     season['pts_diff'] = season['pts_win'].astype(int) - season['pts_lose'].astype(int)

#     # This is a team level dataframe
#     # I append winners to losers to get all possible teams
#     #team_ref = pd.DataFrame(season['winner'].append(season['loser']),columns=['team']).drop_duplicates().set_index(['team']).sort_index()
#     team_ref = pd.DataFrame(pd.concat([season['winner'],season['loser']]),columns=['team']).drop_duplicates().set_index(['team']).sort_index()

#     #initialize vars


#     # Typed these values in from 538.com
#     # teams in alphabetical order
#     if input_season == 2023:

#       elo_list = [
#       [    1320], #Arizona Cardinals
#       [    1480], #Atlanta Falcons
#       [    1600], #Baltimore Ravens
#       [    1675], #Buffalo Bills
#       [    1400], #Carolina Panthers
#       [    1375], #Chicago Bears
#       [    1730], #Cincinatti Bengals
#       [    1575], #Cleveland Browns
#       [    1575], #Dallas Cowboys
#       [    1475], #Denver Broncos
#       [    1575], #Detroit Lions
#       [    1450], #Green Bay 
#       [    1391], #Houston Texans
#       [    1319], #Indianapolis Colts
#       [    1550], #Jax Jaguars
#       [    1705], #KC Chiefs
#       [    1400], #LV Raiders
#       [    1580], #LA Chargers
#       [    1410], #LA Rams
#       [    1525], #Miami Dolphins
#       [    1550], #Minn Vikings
#       [    1500], #NE Patriots
#       [    1504], #NO Saints
#       [    1475], #NY Giants
#       [    1545], #NY Jets
#       [    1730], #Philly Eagles
#       [    1450], #Pitt Steelers
#       [    1575], #49ers
#       [    1550], #Seahawks
#       [    1440], #TB Buccaneers 
#       [    1450], #Tenn Titans 
#       [    1465]] #WSH Commanders      
#     else:
#       elo_list = [
#       [    1475], #Arizona Cardinals
#       [    1500], #Atlanta Falcons
#       [    1750], #Baltimore Ravens
#       [    1725], #Buffalo Bills
#       [    1315], #Carolina Panthers
#       [    1439], #Chicago Bears
#       [    1614], #Cincinatti Bengals
#       [    1520], #Cleveland Browns
#       [    1621], #Dallas Cowboys
#       [    1481], #Denver Broncos
#       [    1700], #Detroit Lions
#       [    1560], #Green Bay 
#       [    1580], #Houston Texans
#       [    1500], #Indianapolis Colts
#       [    1510], #Jax Jaguars
#       [    1742], #KC Chiefs
#       [    1415], #LV Raiders
#       [    1440], #LA Chargers
#       [    1516], #LA Rams
#       [    1590], #Miami Dolphins
#       [    1420], #Minn Vikings
#       [    1370], #NE Patriots
#       [    1440], #NO Saints
#       [    1425], #NY Giants
#       [    1510], #NY Jets
#       [    1615], #Philly Eagles
#       [    1450], #Pitt Steelers
#       [    1720], #49ers
#       [    1550], #Seahawks
#       [    1500], #TB Buccaneers 
#       [    1400], #Tenn Titans 
#       [    1400]] #WSH Commanders


#     team_ref['elo'] = elo_list

#     # Old code to start every team at 1500
#     #team_ref['elo'] = [[1500] for _ in range(len(team_ref))]


#     # Initialize wins
#     team_ref['wins'] = 0
#     team_ref['losses'] = 0


#     # Initialize ELO rating day of the match
#     season['winner_elo'] = 0
#     season['loser_elo'] = 0
#     season['elo_diff'] = 0

#     # Initialize ELO rating adjusted for the given match results
#     season['winner_adj_elo'] = 0
#     season['loser_adj_elo'] = 0
#     season['elo_adj_diff'] = 0


#     # Change the Elo of a team using the index (index is the team name)
#     import math

#     K = 20 # this is the ELO adjustment constant


#     # Iterate through results of the season



#     for i in range(len(season)):

#         # Names of teams that won and lost for a given game
#         winner = season.loc[i]['winner']
#         loser = season.loc[i]['loser']
#         pts_diff = season.loc[i]['pts_diff']


#         # Update counter on team sheet
#         team_ref.at[winner,'wins'] += 1
#         team_ref.at[loser,'losses'] += 1


#         # Set starting ELO

#         season.at[i,'winner_elo'] = team_ref.at[winner,'elo'][-1]
#         season.at[i,'loser_elo'] = team_ref.at[loser,'elo'][-1]
#         season.at[i,'elo_diff'] = season.at[i,'winner_elo'] - season.at[i,'loser_elo']

#         # Calculate Adjusted ELO
#         # https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
#         trans_winner_rating = 10**(season.at[i,'winner_elo'] / 400)
#         trans_loser_rating = 10**(season.at[i,'loser_elo'] / 400)

#         #    print(trans_winner_rating)
#         # print(trans_loser_rating)

#         expected_winner_score = trans_winner_rating / (trans_winner_rating + trans_loser_rating)
        
#         if pts_diff == 0:
#             elo_adj = np.log(abs(pts_diff)+.01) * K * (1 - expected_winner_score)
#         else:
#             elo_adj = np.log(abs(pts_diff)) * K * (1 - expected_winner_score)

#         #expected_loser_score = trans_loser_rating / (trans_winner_rating + trans_loser_rating)

#         season.at[i,'winner_adj_elo'] = season.at[i,'winner_elo'] + elo_adj
#         season.at[i,'loser_adj_elo'] = season.at[i,'loser_elo'] - elo_adj
#         season.at[i,'elo_adj_diff'] = season.at[i,'winner_adj_elo'] - season.at[i,'loser_adj_elo']

#         # Add our new elo scores to the team level spreadsheet

#         team_ref.at[winner,'elo'].append(season.at[i,'winner_adj_elo'])
#         team_ref.at[loser,'elo'].append(season.at[i,'loser_adj_elo'])

#         #team_ref.loc[team_ref.loc[winner], 'wins'] += 1
        
#     # Adds a given value to an elo rating
#     #team_ref.at['New York Giants','elo'].append(team_ref.at['New York Giants','elo'][-1] + 5)
        
    
#     #team_ref['elo'][-1]

    
#     # Get the current ELO, it's the last one in the ELO column list for each team
#     team_ref['Current ELO'] = [ a[-1] for a in team_ref['elo'] ]
    
#     return team_ref['Current ELO'],team_ref
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
df.loc[('DAL', ...)]['posteam_score']['max']



data = pd.DataFrame({'epa':None, 'pass_ypa':None, 'rush_ypa':None, 'turnovers':None, 'takeaways':None,'home_away':None, 'elo':None,
              'pressures_allowed':None, 'total_plays':None, 'points':None,
              
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

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

X_train = train_df.drop(['points','team'],axis=1)
y_train = train_df.points
X_test = test_df.drop(['points','team'],axis=1)
y_test = test_df.points




inputs = tf.keras.Input(shape=(len(X_train.columns),), name = 'input')
hidden1 = tf.keras.layers.Dense(units = 500, activation = 'relu', name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units = 200, activation = 'relu', name = 'hidden2')(hidden1)
dropout1 = tf.keras.layers.Dropout(rate = .2)(hidden2)
hidden3 = tf.keras.layers.Dense(units = 50, activation = 'relu', name = 'hidden3')(dropout1)
dropout2 = tf.keras.layers.Dropout(rate = .2)(hidden3)
hidden4 = tf.keras.layers.Dense(units = 15, activation = 'relu', name = 'hidden4')(dropout2)
dropout3 = tf.keras.layers.Dropout(rate = .2)(hidden4)
hidden5 = tf.keras.layers.Dense(units = 5, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L1(.01), name = 'hidden5')(dropout3)
output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden5)


#hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
model = 0
model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(loss = 'mae', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))# metrics = ['mae'])
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=33)

history = model.fit(x = X_train,y=y_train, validation_split=.15,callbacks = [stop_early],  batch_size = 1, epochs = 100)

















# for game in range(len(s)):
    
#     # home_team_input = f"{s['games'][game]['homeTeam']['fullName']}"
#     # away_team_input = f"{s['games'][game]['awayTeam']['fullName']}"
#     home_abbr = s.loc[s['game_id'] == gid[game], 'home_team'].values[0]
#     away_abbr = s.loc[s['game_id'] == gid[game], 'away_team'].values[0]
#     home_team_input = f"{teams.loc[teams['team_abbr'] == home_abbr,'team_name'].values[0]}"
#     away_team_input = f"{teams.loc[teams['team_abbr'] == away_abbr,'team_name'].values[0]}"

#     # home_team_input = "New York Giants"
#     # away_team_input = "Dallas Cowboys"


#     # hist_elo = pd.read_csv('nfl_historical_elo.csv')
#     # hist_elo = hist_elo[hist_elo.season > 2020]

#     # ############################################################### data preparation
    
#     dfh = nfl_dev().season(team=f'{home_team_input}',data=pbp)
                                            

#     data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
#     data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']


#     off=dfh[dfh['posteam']==home_abbr]
#     defense=dfh[dfh['defteam']==home_abbr]


#     df_o = off[data_sum_off].groupby(by=['game_id']).sum()
#     df_d = defense[data_sum_def].groupby(by=['game_id']).sum()
#     df_d = df_d.rename(columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )


#     for GID in df_o.index:

#         df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]
#         if all(off.loc[off.game_id == GID,'home_team'] == home_abbr):
#             df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
#             df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
#             df_o.loc[GID,'home_away'] = 1
#             df_o.loc[GID,'elo'] = 1
            

#         else:
#             df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
#             df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
#             df_o.loc[GID,'home_away'] = 0



#     # for GID in df_o.index:
#     #     df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]



#     ht_df = pd.concat([df_o,df_d],axis=1)

#     ht_df['season']=None

#     for i in range(len(ht_df)):
#         if ht_df.iloc[i].name[0:4] == '2023':
#             ht_df['season'].iloc[i] = 2023
#         else: 
#             ht_df['season'].iloc[i] = 2024

#     elo_temp_home_2023 = pd.Series(list(team_ref_2023.loc[home_team_input,'elo']))
#     elo_temp_home_2024 = pd.Series(list(team_ref_2024.loc[home_team_input,'elo']))

#     ht_df['elo']=None

#     counter=0
#     for i in range(len(ht_df[ht_df['season'] == 2023])):
#         if i < (len(elo_temp_home_2023)-1):
#             ht_df.loc[ ht_df[ht_df.season == 2023].index[i],'elo'] = elo_temp_home_2023[i]
#         else: 
#             ht_df.loc[ ht_df[ht_df.season == 2023].index[i],'elo'] = team_ref_2023.loc[home_team_input,'Current ELO'] + counter*20
#             counter+=1


#     for i in range(len(ht_df[ht_df['season'] == 2024])):
#         try:ht_df.loc[ ht_df[ht_df.season == 2024].index[i],'elo'] = elo_temp_home_2024[i]
#         except: ht_df.loc[ ht_df[ht_df.season == 2024].index[i],'elo'] = team_ref_2024.loc[home_team_input,'Current ELO']






#     # ############################################################### data preparation
   

#     dfa = nfl_dev().season(team=f'{away_team_input}',data=pbp)
    

#     data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
#     data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']

#     off=dfa[dfa['posteam']==away_abbr]


#     defense=dfa[dfa['defteam']==away_abbr]

#     df_o = off[data_sum_off].groupby(by=['game_id']).sum()
#     df_o['points_scored'] = 0
#     df_d['points_allowed'] = 0

#     df_d = defense[data_sum_def].groupby(by=['game_id']).sum()
#     df_d = df_d.rename(columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )


#     for GID in df_o.index:

#         df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]
#         if all(off.loc[off.game_id == GID,'home_team'] == away_abbr):
#             df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
#             df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
#             df_o.loc[GID,'home_away'] = 1
#         else:
#             df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
#             df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
#             df_o.loc[GID,'home_away'] = 0




#     at_df = pd.concat([df_o,df_d],axis=1)

#     at_df['season']=None

#     for i in range(len(at_df)):
#         if at_df.iloc[i].name[0:4] == '2023':
#             at_df['season'].iloc[i] = 2023
#         else: 
#             at_df['season'].iloc[i] = 2024

#     elo_temp_away_2023 = pd.Series(list(team_ref_2023.loc[away_team_input,'elo']))
#     elo_temp_away_2024 = pd.Series(list(team_ref_2023.loc[away_team_input,'elo']))

#     at_df['elo']=None

#     counter=0
#     for i in range(len(at_df[at_df['season'] == 2023])):
#         if i < (len(elo_temp_away_2023)-1):
#             at_df.loc[ at_df[at_df.season == 2023].index[i],'elo'] = elo_temp_away_2023[i]
#         else: 
#             at_df.loc[ at_df[at_df.season == 2023].index[i],'elo'] = team_ref_2023.loc[away_team_input,'Current ELO'] + counter*20
#             counter+=1



#     for i in range(len(at_df[at_df['season'] == 2024])):
#         try:at_df.loc[ at_df[at_df.season == 2024].index[i],'elo'] = elo_temp_away_2024[i]
#         except: at_df.loc[ at_df[at_df.season == 2024].index[i],'elo'] = team_ref_2024.loc[away_team_input,'Current ELO']




        
        
        
        
#     at_df.elo = at_df.elo.astype('float')
#     ht_df.elo = ht_df.elo.astype('float')  

#     ht_df['turnovers'] = ht_df.fumble_lost + ht_df.interception
#     at_df['turnovers'] = at_df.fumble_lost + at_df.interception
#     ht_df['takeaways'] = ht_df.fumble_gained + ht_df.interceptions_gained
#     at_df['takeaways'] = at_df.fumble_gained + at_df.interceptions_gained











#     ################## home #####################
#     #feature = ['epa', 'epa_allowed', 'passing_yards', 'rushing_yards', 'turnovers', 'takeaways','home_away', 'elo','weights']
#     predictors = ['epa', 'epa_allowed', 'passing_yards', 'rushing_yards', 'turnovers', 'takeaways','home_away', 'elo']
#     y = ht_df['points_scored']
#     weights = np.ones(len(ht_df))
#     for i in range(len(weights)-1):
#         weights[i+1] = weights[i] + .25
    
#     # ht_df['weights'] = weights        



#     # X, y = ht_df[predictors],  ht_df[['points_scored','weights']]



#     # params = {"objective": "reg:absoluteerror", "tree_method": "gpu_hist"}
#     # # Create regression matrices
#     # dtrain_reg = xgb.DMatrix(X[predictors], y['points_scored'],weight =y['weights'] )
#     # #dtest_reg =xgb.DMatrix(X_test[predictors], y_test['points_scored'],weight =y_test['weights'] )
    
#     # n = 50
#     # model_home = xgb.train(
#     # params=params,
#     # dtrain=dtrain_reg,
#     # num_boost_round=n,
#     # verbose_eval=25
#     # )
  
    
#     model_home = LinearRegression().fit(ht_df[predictors],y,sample_weight=weights)


#     exp_epa = (stats.median(ht_df[len(ht_df)-10:len(ht_df)].epa) + stats.median(at_df[len(at_df)-10:len(at_df)].epa_allowed))/2
#     exp_epa_allowed = (stats.median(ht_df[len(ht_df)-10:len(ht_df)].epa_allowed) + stats.median(at_df[len(at_df)-10:len(at_df)].epa))/2
#     exp_passing_yards = (stats.median(ht_df[len(ht_df)-10:len(ht_df)].passing_yards) + stats.median(at_df[len(at_df)-10:len(at_df)].passing_yards_allowed))/2
#     exp_rushing_yards = (stats.median(ht_df[len(ht_df)-10:len(ht_df)].rushing_yards) + stats.median(at_df[len(at_df)-10:len(at_df)].rushing_yards_allowed))/2
#     exp_turnovers = (stats.mean(ht_df[len(ht_df)-10:len(ht_df)].turnovers) + stats.mean(at_df[len(at_df)-10:len(at_df)].takeaways))/2
#     exp_takeaways = (stats.mean(ht_df[len(ht_df)-10:len(ht_df)].takeaways) + stats.mean(at_df[len(at_df)-10:len(at_df)].turnovers))/2

#     home_score_proj = model_home.predict(np.array([[exp_epa,exp_epa_allowed,exp_passing_yards,
#                                                     exp_rushing_yards,exp_turnovers,
#                                                     exp_takeaways,1, team_ref_2024.loc[home_team_input,'Current ELO']]]))[0]
#     #obj  = pd.DataFrame({'epa':exp_epa, 'epa_allowed':exp_epa_allowed, 'passing_yards':exp_passing_yards, 'rushing_yards':exp_rushing_yards, 'turnovers':exp_turnovers, 'takeaways':exp_takeaways,'home_away':[1], 'elo':team_ref_2022.loc[home_team_input,'Current ELO']})
    
#     #home_score_proj = model_home.predict(xgb.DMatrix(obj))[0]
    
    






#     ################## away #####################


#     predictors = ['epa', 'epa_allowed', 'passing_yards', 'rushing_yards', 'turnovers', 'takeaways','home_away', 'elo']
#     y = at_df['points_scored']
#     weights = np.ones(len(at_df))
#     for i in range(len(weights)-1):
#         weights[i+1] = weights[i] + .5
#     model_away = LinearRegression().fit(at_df[predictors],y, sample_weight=weights)
#     #at_df['weights'] = weights         
#     #model_home = LinearRegression().fit(ht_df[predictors],y,sample_weight=weights)


#     exp_epa = (stats.median(at_df[len(at_df)-10:len(at_df)].epa) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].epa_allowed))/2
#     exp_epa_allowed = (stats.median(at_df[len(at_df)-10:len(at_df)].epa_allowed) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].epa))/2
#     exp_passing_yards = (stats.median(at_df[len(at_df)-10:len(at_df)].passing_yards) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].passing_yards_allowed))/2
#     exp_rushing_yards = (stats.median(at_df[len(at_df)-10:len(at_df)].rushing_yards) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].rushing_yards_allowed))/2
#     exp_turnovers = (stats.mean(at_df[len(at_df)-10:len(at_df)].turnovers) + stats.mean(ht_df[len(ht_df)-10:len(ht_df)].takeaways))/2
#     exp_takeaways = (stats.mean(at_df[len(at_df)-10:len(at_df)].takeaways) + stats.mean(ht_df[len(ht_df)-10:len(ht_df)].turnovers))/2


#     away_score_proj = model_away.predict(np.array([[exp_epa,exp_epa_allowed,exp_passing_yards,exp_rushing_yards,exp_turnovers,exp_takeaways,0, team_ref_2024.loc[away_team_input,'Current ELO']]]))[0]

#     #obj  = pd.DataFrame({'epa':exp_epa, 'epa_allowed':exp_epa_allowed, 'passing_yards':exp_passing_yards, 'rushing_yards':exp_rushing_yards, 'turnovers':exp_turnovers, 'takeaways':exp_takeaways,'home_away':[0], 'elo':team_ref_2022.loc[away_team_input,'Current ELO']})
    
#     #away_score_proj = model_away.predict(xgb.DMatrix(obj))[0]

    
#     # exp_epa = (stats.median(at_df[len(at_df)-10:len(at_df)].epa) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].epa_allowed))/2
#     # exp_epa_allowed = (stats.median(at_df[len(at_df)-10:len(at_df)].epa_allowed) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].epa))/2
#     # exp_passing_yards = (stats.median(at_df[len(at_df)-10:len(at_df)].passing_yards) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].passing_yards_allowed))/2
#     # exp_rushing_yards = (stats.median(at_df[len(at_df)-10:len(at_df)].rushing_yards) + stats.median(ht_df[len(ht_df)-10:len(ht_df)].rushing_yards_allowed))/2
#     # exp_turnovers = (stats.mean(at_df[len(at_df)-10:len(at_df)].turnovers) + stats.mean(ht_df[len(ht_df)-10:len(ht_df)].takeaways))/2
#     # exp_takeaways = (stats.mean(at_df[len(at_df)-10:len(at_df)].takeaways) + stats.mean(ht_df[len(ht_df)-10:len(ht_df)].turnovers))/2





#     df.loc[s['game_id'].iloc[game],'Home_Team'] = f"{s['home_team'].iloc[game]}"
#     df.loc[s['game_id'].iloc[game],'Away_Team'] = f"{s['away_team'].iloc[game]}"
#     df.loc[s['game_id'].iloc[game],'Home_Score'] = home_score_proj
#     df.loc[s['game_id'].iloc[game],'Away_Score'] = away_score_proj
#     df.loc[s['game_id'].iloc[game],'Spread'] = home_score_proj- away_score_proj
#     df.loc[s['game_id'].iloc[game],'Total'] = home_score_proj + away_score_proj



#     print(f"{home_team_input}: {np.mean(home_score_proj)}")
#     print(f"{away_team_input}: {np.mean(away_score_proj)}")







# df






