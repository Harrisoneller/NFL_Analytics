home_team = 'ARI'
away_team = 'BUF'



import numpy as np 
import pandas as pd
import sportsdataverse.nfl as nfl
import scipy
import difflib
import json

try: df = pd.read_excel(r'C:\Users\Harrison Eller\sports-app-server\NFL_Team_Comp_Data.xlsx')
except: df = pd.read_excel(r'C:\Users\harri\sports-app-server\NFL_Team_Comp_Data.xlsx')
df.index = df.Team




import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data

 
# ------- PART 1: Create background
 
# number of variable
feature_cols = ['EPA', 'PROE','YPG','PPG','Plays_per_Game']
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
df = df.rename(columns={'off_epa':"EPA",'proe':'PROE','total_yards':'YPG','total_points':'PPG','total_off_plays':'Plays_per_Game'})
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
plt.legend(loc='upper right', bbox_to_anchor=(0.05, 0.05))
for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')
    
plt.show()








df.columns

########################### away ###############

# ------- PART 1: Create background
 
# number of variable
feature_cols = ['EPA_Allowed', 'PPG_Allowed','YPG_Allowed','Plays_per_Game_Allowed','Turnover_Rate']
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
df = df.rename(columns={'off_epa_allowed':"EPA_Allowed",'total_points_allowed':'PPG_Allowed','total_yards_allowed':'YPG_Allowed','turnovers':'Turnover_Rate','total_def_plays':'Plays_per_Game_Allowed'})
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
plt.legend(loc='upper right', bbox_to_anchor=(0.05, 0.05))
for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')
    
plt.show()