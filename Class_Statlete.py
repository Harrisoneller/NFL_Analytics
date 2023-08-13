# class for pulling WR data
import numpy as np 
import pandas as pd
import sportsdataverse.nfl as nfl
import scipy
import difflib
import json
import datetime
ID = nfl.load_nfl_players()
stats = nfl.load_nfl_player_stats()
ngs = nfl.load_nfl_ngs_receiving()  
    

class Statlete:
    import numpy as np 
    import pandas as pd
    import sportsdataverse.nfl as nfl
    import scipy
    import difflib
    import json
    
    """ Need to adjust season year to 2023 once season starts"""
    ID = nfl.load_nfl_players()
    stats = nfl.load_nfl_player_stats()
    ngs = nfl.load_nfl_ngs_receiving()  
    
    #### init ####
    def __init__(self, player):
        player = difflib.get_close_matches(player, ID.display_name.unique(), n=1, cutoff=0.6)
        player_ngs = difflib.get_close_matches(player, ngs.player_display_name.unique(), n=1, cutoff=0.6)
        self.player = player
        self.player_ngs = player_ngs
        self.game = self.Game(player)
        self.season = self.Season(player)
        self.career = self.Season(player)
        self.ngs = self.NextGenStats(player_ngs)
        
        
    class Game:
        ID = nfl.load_nfl_players()
        stats = nfl.load_nfl_player_stats()
        ngs = nfl.load_nfl_ngs_receiving() 
        
        def __init__(self,player):
            #player = difflib.get_close_matches(player, ID.display_name.unique(), n=1, cutoff=0.6)
            player_ngs = difflib.get_close_matches(player, ngs.player_display_name.unique(), n=1, cutoff=0.6)
            self.player_game = player
            self.player_ngs = player_ngs
                    
        ############################ get data on a game to game basis ########################
        def get_data(self, week, season = 2022):
            stats = nfl.load_nfl_player_stats()  
            player_df = stats[stats.player_display_name.isin(self.player_game)]
            output1 = player_df[player_df['week'] == week]
            try:
                df = output1[output1['season'] == season]
            except: 
                df = output1[output1['season'].isin(season)]
            return df.to_json(orient = 'columns' )
               
        
        
    class Season:       
        ID = nfl.load_nfl_players()
        stats = nfl.load_nfl_player_stats()
        ngs = nfl.load_nfl_ngs_receiving() 
        
        def __init__(self, player):
            #player_ngs = difflib.get_close_matches(player, ngs.player_display_name.unique(), n=1, cutoff=0.6)
            #self.player = super().__init__(player)
            self.player_season = player
            #self.player_ngs = player_ngs 

        def get_data(self, season = [2022]):
            stats = nfl.load_nfl_player_stats()  
            player_df = stats[stats.player_display_name.isin(self.player_season)]
            #output1 = player_df[player_df['week'] == week]
            try:
                df = player_df[player_df['season'].isin(season)]
            except: 
                df = player_df[player_df['season'] == season[0]]
            return df.to_json(orient = 'columns' )

    class Career:       
        ID = nfl.load_nfl_players()
        stats = nfl.load_nfl_player_stats()
        ngs = nfl.load_nfl_ngs_receiving() 
        
        def __init__(self,player):
            #player = difflib.get_close_matches(player, ID.display_name.unique(), n=1, cutoff=0.6)
            #player_ngs = difflib.get_close_matches(player, ngs.player_display_name.unique(), n=1, cutoff=0.6)
            self.player_career =  player
            #self.player_ngs = player_ngs 

        def get_data(self, season = stats.season.unique()):
            stats = nfl.load_nfl_player_stats()  
            player_df = stats[stats.player_display_name.isin(self.player_career)]
            #output1 = player_df[player_df['week'] == week]
            try:
                df = player_df[player_df['season'].isin(season)]
            except: 
                df = player_df[player_df['season'] == season[len(season)]]
            return df.to_json(orient = 'columns' )
        
        
    class NextGenStats:
        ID = nfl.load_nfl_players()
        stats = nfl.load_nfl_player_stats()
        ngs = nfl.load_nfl_ngs_receiving() 
        
        def __init__(self,player):
            #player = difflib.get_close_matches(player, ID.display_name.unique(), n=1, cutoff=0.6)
            self.player = player
            #self.player = player_ngs
            
        def get_data(self, season = [2022], week = False):
            ngs = nfl.load_nfl_ngs_receiving() 
            player_df = ngs[ngs.player_display_name.isin(self.player_ngs)]
            
            if len(season) > 1:
                df = player_df[player_df['season'].isin(season)]
            elif len(season) == 1: 
                df = player_df[player_df['season'] == season[0]]
                
            if week != False: 
                df = player_df[player_df['week'] == week]
            return df.to_json(orient = 'columns' )
        
    class PlayerInfo:
        ID = nfl.load_nfl_players()
        stats = nfl.load_nfl_player_stats()
        ngs = nfl.load_nfl_ngs_receiving() 

        def __init__(self,player):
            self.player =player
            #self.player_ngs = player_ngs
            
        def get_info(self, season = 2022.0): ############ season input format is different here
            ID = nfl.load_nfl_players()
            df = ID[ID['display_name'].isin(self.player)]
            ### filter for current season ###    
            #df =  player_df[player_df['season'] == season]
            return df.to_json(orient = 'columns')
















####################################### Team #########################################################


class Team:
    import numpy as np 
    import pandas as pd
    import sportsdataverse.nfl as nfl
    import scipy
    import difflib
    import json

    #teams_info = nfl.nfl_loaders.load_nfl_teams()

    def __init__(self, team):
        teams_info = nfl.nfl_loaders.load_nfl_teams()
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
            teams_info = nfl.nfl_loaders.load_nfl_teams()
            team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
            self.team_season = team
            self.team_abr_season = teams_info.team_abbr[teams_info.team_name.isin(team)]
            #self.player_ngs = player_ngs 

        def get_data(self, season = [2022]):
            pbp = nfl.load_nfl_pbp(seasons=(season))
            team_pbp = pbp[(pbp.home_team.isin(self.team_abr)) | (pbp.away_team.isin(self.team_abr))]
            team_off_pbp = team_pbp[team_pbp.posteam.isin(self.team_abr)]
             
            
            #output1 = player_df[player_df['week'] == week]
            try:
                df = team_off_pbp
            except:
                df = team_off_pbp
            return df
        
        
        
#Team.Season('Falcons').get_data()