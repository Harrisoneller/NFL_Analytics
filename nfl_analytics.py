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

        def get_data(self, season = [2022,2023],data = False):
            if data is not False:
                print("data received")
                pbp = data
            else:
                pbp = nfl.load_nfl_pbp(seasons=(season))
            team_pbp = pbp[(pbp.home_team.isin(self.team_abr_season)) | (pbp.away_team.isin(self.team_abr_season))]
            team_off_pbp = team_pbp[(team_pbp.posteam.isin(self.team_abr_season)) | (team_pbp.defteam.isin(self.team_abr_season)) ]
            
            
            #output1 = player_df[player_df['week'] == week]  
            try:
                df = team_off_pbp
            except:
                df = team_off_pbp
            return df
  



pbp = nfl.load_nfl_pbp(seasons=([2023]),return_as_pandas=True)
pbp = pbp[pbp.season_type == "REG"]





