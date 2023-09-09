from Class_Statlete import Statlete
import sportsdataverse.nfl as nfl 
import pandas as pd
import numpy as np 


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








df_pass = nfl.nfl_loaders.load_nfl_ngs_passing()
#df=df[df['season'] == 2022]

df_pass.to_csv('qb_ngs.csv')


df_rushing = nfl.nfl_loaders.load_nfl_ngs_rushing()
#df=df[df['season'] == 2022]

df_rushing.to_csv('rb_ngs.csv')


df_receiving = nfl.nfl_loaders.load_nfl_ngs_receiving()
#df=df[df['season'] == 2022]

df_receiving.to_csv('receiving_ngs.csv')



stats = nfl.nfl_loaders.load_nfl_player_stats()


#stats[stats['player_display_name'] == 'Josh Allen']
#qb[qb['player_display_name'] == 'Josh Allen'].player_gsis_id

passing = pd.merge(stats, df_pass, on=['player_display_name','season','week'])
receiving = pd.merge(stats,df_receiving, on=['player_display_name','season','week'])
rushing = pd.merge(stats,df_rushing, on=['player_display_name','season','week'])


passing.to_csv('passing.csv')
receiving.to_csv('receving.csv')
rushing.to_csv('rushing.csv')




passing = passing[passing['season'] > 2017]

test = passing[passing.player_display_name == 'Jared Goff']
test.passing_yards.hist()






