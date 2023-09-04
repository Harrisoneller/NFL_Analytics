home_team = 'ARI'
away_team = 'BUF'



import numpy as np 
import pandas as pd
import sportsdataverse.nfl as nfl
import scipy
import difflib
import json

df = pd.read_excel(r'C:\Users\Harrison Eller\sports-app-server\NFL_Team_Comp_Data.xlsx')
df.index = df.Team
# class Team:
#     import numpy as np 
#     import pandas as pd
#     import sportsdataverse.nfl as nfl
#     import scipy
#     import difflib
#     import json

#     #teams_info = nfl.nfl_loaders.load_nfl_teams()

#     def __init__(self, team):
#         teams_info = nfl.nfl_loaders.load_nfl_teams()
#         team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
#         self.team = team
#         self.team_abr = list(teams_info.team_abbr[teams_info.team_name.isin(self.team)])[0]
#         #self.game = self.Game(team)
#         self.season = self.Season(team)
    
        
        
#     class Season:       
#         ID = nfl.load_nfl_players()
#         stats = nfl.load_nfl_player_stats()
#         ngs = nfl.load_nfl_ngs_receiving() 
        
#         def __init__(self, team):
#             teams_info = nfl.nfl_loaders.load_nfl_teams()
#             team = difflib.get_close_matches(team, teams_info.team_name.unique(), n=1, cutoff=0.6)
#             self.team_season = team
#             self.team_abr_season = teams_info.team_abbr[teams_info.team_name.isin(team)]
#             #self.player_ngs = player_ngs 

#         def get_data(self, season = [2022]):
#             pbp = nfl.load_nfl_pbp(seasons=(season))
#             team_pbp = pbp[(pbp.home_team.isin(self.team_abr_season)) | (pbp.away_team.isin(self.team_abr_season))]
#             team_off_pbp = team_pbp[team_pbp.posteam.isin(self.team_abr_season)]
             
            
#             #output1 = player_df[player_df['week'] == week]
#             try:
#                 df = team_off_pbp
#             except:
#                 df = team_off_pbp
#             return df
        
        
#         def get_data_full(self, season = [2022]):
#             pbp = nfl.load_nfl_pbp(seasons=(season))
#             team_pbp = pbp[(pbp.home_team.isin(self.team_abr_season)) | (pbp.away_team.isin(self.team_abr_season))]
#             team_pbp = team_pbp[team_pbp.down.isin([1,2,3,4])]
#             #team_off_pbp = team_pbp[team_pbp.posteam.isin(self.team_abr_season)]
             
            
#             #output1 = player_df[player_df['week'] == week]
#             try:
#                 df = team_pbp
#             except:
#                 df = team_pbp
#             return df      
        



# pbp = nfl.load_nfl_pbp(seasons=([2022]))
# team_info = nfl.load_nfl_teams()

# pbp.columns[0:30]

# ###### calculate offensive success rate ########
# pbp['OFF_SR'] = 0


# pbp.OFF_SR[ (pbp['down'] == 1) & ((pbp['yards_gained']/pbp['ydstogo']) >=.5)] = 1
# pbp.OFF_SR[ (pbp['down'] == 2) & ((pbp['yards_gained']/pbp['ydstogo']) >=.7)] = 1
# pbp.OFF_SR[ (pbp['down'] == 3) & ((pbp['yards_gained']/pbp['ydstogo']) >=.99)] = 1
# pbp.OFF_SR[ (pbp['down'] == 4) & ((pbp['yards_gained']/pbp['ydstogo']) >=.99)] = 1
# pbp_down_filter = pbp[pbp['down'].isin([1,2,3,4])]
# pbp_down_filter['play_count'] = 1
# pbp_df = pbp_down_filter[['passing_yards','posteam','rushing_yards','air_yards','OFF_SR','play_count']].groupby(by=['posteam']).sum()
# pbp_df[pbp_df.columns[0:3]] = pbp_df[pbp_df.columns[0:3]]/17
# pbp_df['OSR'] = pbp_df['OFF_SR']/pbp_df['play_count']
# pbp_df['OSR'] = pbp_df['OSR'].rank()
# data_offense = pbp[['passing_yards','posteam','rushing_yards','air_yards','posteam_score','epa','game_id']].groupby(by=['posteam']).mean()
# data_offense
# pbp_df['epa'] =  data_offense['epa']
# pbp_df= pbp_df.rank(ascending=False)
# pbp_df['team'] = pbp_df.index 

# # tot_data_offense_sum = pbp[['passing_yards','posteam','rushing_yards']].groupby(by=['posteam']).sum()


# # ### per game summation
# # data_offense_sum = pbp[['passing_yards','posteam','rushing_yards','air_yards','game_id']].groupby(by=['posteam']).sum()
# # data_offense_sum = data_offense_sum/17
# # data_offense_sum['team'] = data_offense_sum.index


# ### get epa 


# #### get rank
# #data_offense_sum = data_offense_sum.rank(ascending=False)






import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data

 
# ------- PART 1: Create background
 
# number of variable
feature_cols = ['total_points', 'total_yards','off_epa','proe','total_off_plays']
categories=list(df)
#categories = categories[0:3] + [categories[len(categories)-3], categories[len(categories)-2]]
categories = feature_cols
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
#plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
#plt.yticks([100,200,300], ["10","20","30"], color="grey", size=7)
plt.ylim(32,1)
 

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
home_team = 'ATL'
away_team = 'CAR'
values=df.loc[f'{home_team}',feature_cols].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label= f'{home_team}')
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[f'{away_team}',feature_cols].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'{away_team}')
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()



