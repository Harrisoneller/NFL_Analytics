
from Class_Statlete import Statlete
import numpy as np 
import pandas as pd
import sportsdataverse.nfl as nfl
import scipy
import difflib
import json
import numpy as np
#import matplotlib.pyplot as plt
import time


class Team:
    import numpy as np 
    import pandas as pd
    import sportsdataverse.nfl as nfl
    import scipy
    import difflib
    import json

    #teams_info = nfl.nfl_loaders.load_nfl_teams()

    def __init__(self,team):
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

        def get_data(self, season = [2019,2022]):
            pbp = nfl.load_nfl_pbp(seasons=(season))
            team_pbp = pbp[(pbp.home_team.isin(self.team_abr_season)) | (pbp.away_team.isin(self.team_abr_season))]
            team_off_pbp = team_pbp[(team_pbp.posteam.isin(self.team_abr_season)) | (team_pbp.defteam.isin(self.team_abr_season)) ]
            
            
            #output1 = player_df[player_df['week'] == week]  
            try:
                df = team_off_pbp
            except:
                df = team_off_pbp
            return df
  
home_team_input = 'Atlanta Falcons'
away_team_input = 'Carolina Panthers'     
# hist_elo = pd.read_csv('nfl_historical_elo.csv')
# hist_elo = hist_elo[hist_elo.season > 2020]

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

data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']

off=dfh[dfh['posteam']==Team(f'{home_team_input}').team_abr]


defense=dfh[dfh['defteam']==Team(f'{home_team_input}').team_abr]

df_o = off[data_sum_off].groupby(by=['game_id']).sum()

GID = '2022_18_TB_ATL'
for GID in df_o.index:
    df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]


df_d = defense[data_sum_def].groupby(by=['game_id']).sum()
df_d = df_d.rename(columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )

ht_df = pd.concat([df_o,df_d],axis=1)









# ############################################################### data preparation
TEAM = Team(f'{away_team_input}')
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

data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']

off=dfh[dfh['posteam']==Team(f'{home_team_input}').team_abr]


defense=dfh[dfh['defteam']==Team(f'{home_team_input}').team_abr]

df_o = off[data_sum_off].groupby(by=['game_id']).sum()
for GID in df_o.index:

    df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]
    if any(off.loc[off.game_id == GID,'home_team'] == Team(f'{home_team_input}').team_abr):
      df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_home_score'])
      df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_away_score'])
    else:
      df_d.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_home_score'])
      df_o.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_away_score'])

for GID in df_o.index:
    df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]


df_d = defense[data_sum_def].groupby(by=['game_id']).sum()
df_d = df_d.rename(columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )

at_df = pd.concat([df_o,df_d],axis=1)



import scipy
import statistics as stats

#################### home team ################################################




exp_passing_yards = (stats.median(ht_df.passing_yards) + stats.median(at_df.passing_yards_allowed))/2
exp_rushing_yards = (stats.median(ht_df.rushing_yards) + stats.median(at_df.rushing_yards_allowed))/2
exp_interceptions = (stats.mean(ht_df.interception) + stats.mean(at_df.interceptions_gained))/2
exp_fumbles = (stats.mean(ht_df.fumble_lost) + stats.mean(at_df.fumble_gained))/2

