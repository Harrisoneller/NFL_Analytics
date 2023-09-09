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

# Calculate ELO rating for NFL teams using data scraped from web
# Starting out with calculating # of wins

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
  
    
def get_current_elo(input_season):
    
    # Data source we are going to scrape for results
    data_url = f'https://www.pro-football-reference.com/years/{input_season}/games.htm#games::none'
            

    # Scrape the index of a given page
    # Return a list of specified web elements
    def scrape(selection,parent_object,element_type):

        
        # Select the given div
    #  data = soup.findAll("div", { "class" : "table_outer_container" })
        
        list_links = []
        data = soup.findAll(parent_object, { "data-stat" : selection })
        for element in data:
            #print(element['href'])
            
            # Add NA option 
            if element_type!='na':          
                list_links += [a.contents[0] for a in element.findAll(element_type)]
            else:
                # Extracts number if it exists
                if str(element.renderContents()) != "b''":
                    list_links += [str(element.renderContents()).split('\'')[1]]
                    
            #print(list_links)
    # 
        return list_links



    # This is the web data we 
    page = requests.get(data_url)
    soup = BeautifulSoup(page.content, 'html.parser')


    # Automatically only goes as far as shortest list
    # which is the pts_win (limits to only current games played)
    import numpy as np
    # this is a game level dataframe

    length = len(scrape("pts_win",'td','strong'))

    week = scrape("week_num",'th','na')

    # Remove all the text from our week data column
    while 'Week' in week: week.remove('Week')

    season = pd.DataFrame(np.column_stack([week[:length],scrape("winner",'td','a')[:length],scrape("loser",'td','a')[:length],scrape("pts_win",'td','strong')[:length],scrape("pts_lose",'td','na')[:length]]),columns=['week','winner','loser',"pts_win",'pts_lose'])

    season['pts_diff'] = season['pts_win'].astype(int) - season['pts_lose'].astype(int)

    # This is a team level dataframe
    # I append winners to losers to get all possible teams
    team_ref = pd.DataFrame(season['winner'].append(season['loser']),columns=['team']).drop_duplicates().set_index(['team']).sort_index()

    #initialize vars


    # Typed these values in from 538.com
    # teams in alphabetical order
    if input_season == 2022:
      elo_list = [
      [    1501], #Arizona Cardinals
      [    1437], #Atlanta Falcons
      [    1509], #Baltimore Ravens
      [    1614], #Buffalo Bills
      [    1411], #Carolina Panthers
      [    1445], #Chicago Bears
      [    1558], #Cincinatti Bengals
      [    1502], #Cleveland Browns
      [    1562], #Dallas Cowboys
      [    1447], #Denver Broncos
      [    1406], #Detroit Lions
      [    1589], #Green Bay 
      [    1410], #Houston Texans
      [    1543], #Indianapolis Colts
      [    1351], #Jax Jaguars
      [    1629], #KC Chiefs
      [    1493], #LV Raiders
      [    1506], #LA Chargers
      [    1615], #LA Rams
      [    1540], #Miami Dolphins
      [    1514], #Minn Vikings
      [    1537], #NE Patriots
      [    1544], #NO Saints
      [    1389], #NY Giants
      [    1365], #NY Jets
      [    1502], #Philly Eagles
      [    1510], #Pitt Steelers
      [    1576], #49ers
      [    1527], #Seahawks
      [    1610], #TB Buccaneers 
      [    1556], #Tenn Titans 
      [    1465]] #WSH Commanders
    else:
      elo_list = [
      [    1320], #Arizona Cardinals
      [    1480], #Atlanta Falcons
      [    1600], #Baltimore Ravens
      [    1675], #Buffalo Bills
      [    1400], #Carolina Panthers
      [    1375], #Chicago Bears
      [    1730], #Cincinatti Bengals
      [    1575], #Cleveland Browns
      [    1575], #Dallas Cowboys
      [    1475], #Denver Broncos
      [    1575], #Detroit Lions
      [    1450], #Green Bay 
      [    1391], #Houston Texans
      [    1319], #Indianapolis Colts
      [    1550], #Jax Jaguars
      [    1705], #KC Chiefs
      [    1400], #LV Raiders
      [    1580], #LA Chargers
      [    1410], #LA Rams
      [    1525], #Miami Dolphins
      [    1550], #Minn Vikings
      [    1500], #NE Patriots
      [    1504], #NO Saints
      [    1475], #NY Giants
      [    1545], #NY Jets
      [    1730], #Philly Eagles
      [    1450], #Pitt Steelers
      [    1575], #49ers
      [    1550], #Seahawks
      [    1440], #TB Buccaneers 
      [    1450], #Tenn Titans 
      [    1465]] #WSH Commanders


    team_ref['elo'] = elo_list

    # Old code to start every team at 1500
    #team_ref['elo'] = [[1500] for _ in range(len(team_ref))]


    # Initialize wins
    team_ref['wins'] = 0
    team_ref['losses'] = 0


    # Initialize ELO rating day of the match
    season['winner_elo'] = 0
    season['loser_elo'] = 0
    season['elo_diff'] = 0

    # Initialize ELO rating adjusted for the given match results
    season['winner_adj_elo'] = 0
    season['loser_adj_elo'] = 0
    season['elo_adj_diff'] = 0


    # Change the Elo of a team using the index (index is the team name)
    import math

    K = 20 # this is the ELO adjustment constant


    # Iterate through results of the season



    for i in range(len(season)):

        # Names of teams that won and lost for a given game
        winner = season.loc[i]['winner']
        loser = season.loc[i]['loser']
        pts_diff = season.loc[i]['pts_diff']


        # Update counter on team sheet
        team_ref.at[winner,'wins'] += 1
        team_ref.at[loser,'losses'] += 1


        # Set starting ELO

        season.at[i,'winner_elo'] = team_ref.at[winner,'elo'][-1]
        season.at[i,'loser_elo'] = team_ref.at[loser,'elo'][-1]
        season.at[i,'elo_diff'] = season.at[i,'winner_elo'] - season.at[i,'loser_elo']

        # Calculate Adjusted ELO
        # https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
        trans_winner_rating = 10**(season.at[i,'winner_elo'] / 400)
        trans_loser_rating = 10**(season.at[i,'loser_elo'] / 400)

        #    print(trans_winner_rating)
        # print(trans_loser_rating)

        expected_winner_score = trans_winner_rating / (trans_winner_rating + trans_loser_rating)
        
        if pts_diff == 0:
            elo_adj = np.log(abs(pts_diff)+.01) * K * (1 - expected_winner_score)
        else:
            elo_adj = np.log(abs(pts_diff)) * K * (1 - expected_winner_score)

        #expected_loser_score = trans_loser_rating / (trans_winner_rating + trans_loser_rating)

        season.at[i,'winner_adj_elo'] = season.at[i,'winner_elo'] + elo_adj
        season.at[i,'loser_adj_elo'] = season.at[i,'loser_elo'] - elo_adj
        season.at[i,'elo_adj_diff'] = season.at[i,'winner_adj_elo'] - season.at[i,'loser_adj_elo']

        # Add our new elo scores to the team level spreadsheet

        team_ref.at[winner,'elo'].append(season.at[i,'winner_adj_elo'])
        team_ref.at[loser,'elo'].append(season.at[i,'loser_adj_elo'])

        #team_ref.loc[team_ref.loc[winner], 'wins'] += 1
        
    # Adds a given value to an elo rating
    #team_ref.at['New York Giants','elo'].append(team_ref.at['New York Giants','elo'][-1] + 5)
        
    
    #team_ref['elo'][-1]

    
    # Get the current ELO, it's the last one in the ELO column list for each team
    team_ref['Current ELO'] = [ a[-1] for a in team_ref['elo'] ]
    
    return team_ref['Current ELO'],team_ref


out_elo, team_ref_2022= get_current_elo(input_season=2022)
team_ref_2022['season']=None
team_ref_2022['season']=2022
# out_elo, team_ref_2023= get_current_elo(input_season=2023)
# team_ref_2023['season']=None
# team_ref_2023['season']=2023


elo_list = [
      [    1320], #Arizona Cardinals
      [    1480], #Atlanta Falcons
      [    1600], #Baltimore Ravens
      [    1675], #Buffalo Bills
      [    1400], #Carolina Panthers
      [    1375], #Chicago Bears
      [    1730], #Cincinatti Bengals
      [    1575], #Cleveland Browns
      [    1575], #Dallas Cowboys
      [    1475], #Denver Broncos
      [    1575], #Detroit Lions
      [    1450], #Green Bay 
      [    1391], #Houston Texans
      [    1319], #Indianapolis Colts
      [    1550], #Jax Jaguars
      [    1705], #KC Chiefs
      [    1400], #LV Raiders
      [    1580], #LA Chargers
      [    1410], #LA Rams
      [    1525], #Miami Dolphins
      [    1550], #Minn Vikings
      [    1500], #NE Patriots
      [    1504], #NO Saints
      [    1475], #NY Giants
      [    1545], #NY Jets
      [    1730], #Philly Eagles
      [    1450], #Pitt Steelers
      [    1575], #49ers
      [    1550], #Seahawks
      [    1440], #TB Buccaneers 
      [    1450], #Tenn Titans 
      [    1465]] #WSH Commanders






s = nfl.nfl_games.nfl_game_schedule(season = 2023,week=1)
s['games']
gid = [] 
for game in range(len(s['games'])):
    gid.append(s['games'][game]['id'])

df = pd.DataFrame(columns = ['Home_Team', 'Home_Score','Away_Team', 'Away_Score','Spread','Total'], index = gid )





pbp = nfl.load_nfl_pbp(seasons=([2022,2023]))
data=pbp




for game in range(1,len(s['games'])):
    
    home_team_input = f"{s['games'][game]['homeTeam']['fullName']}"
    away_team_input = f"{s['games'][game]['awayTeam']['fullName']}"

    # home_team_input = "New York Giants"
    # away_team_input = "Dallas Cowboys"


    # hist_elo = pd.read_csv('nfl_historical_elo.csv')
    # hist_elo = hist_elo[hist_elo.season > 2020]

    # ############################################################### data preparation
    TEAM = Team(f'{home_team_input}')
    try:
        dfh = Team.Season(f'{home_team_input}').get_data(data=pbp)
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
    df_d = defense[data_sum_def].groupby(by=['game_id']).sum()
    df_d = df_d.rename(columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )


    for GID in df_o.index:

        df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]
        if all(off.loc[off.game_id == GID,'home_team'] == Team(f'{home_team_input}').team_abr):
            df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
            df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
            df_o.loc[GID,'home_away'] = 1
            df_o.loc[GID,'elo'] = 1
            

        else:
            df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
            df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
            df_o.loc[GID,'home_away'] = 0



    # for GID in df_o.index:
    #     df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]



    ht_df = pd.concat([df_o,df_d],axis=1)

    ht_df['season']=None

    for i in range(len(ht_df)):
        if ht_df.iloc[i].name[0:4] == '2022':
            ht_df['season'].iloc[i] = 2022
        else: 
            ht_df['season'].iloc[i] = 2023

    elo_temp_home_2022 = pd.Series(list(team_ref_2022.loc[home_team_input,'elo']))
    #elo_temp_home_2023 = pd.Series(list(team_ref_2023.loc[home_team_input,'elo']))

    ht_df['elo']=None

    counter=0
    for i in range(len(ht_df[ht_df['season'] == 2022])):
        if i < (len(elo_temp_home_2022)-1):
            ht_df.loc[ ht_df[ht_df.season == 2022].index[i],'elo'] = elo_temp_home_2022[i]
        else: 
            ht_df.loc[ ht_df[ht_df.season == 2022].index[i],'elo'] = team_ref_2022.loc[home_team_input,'Current ELO'] + counter*20
            counter+=1


    # for i in range(len(ht_df[ht_df['season'] == 2023])):
    #     try:ht_df.loc[ ht_df[ht_df.season == 2023].index[i],'elo'] = elo_temp_home_2023[i]
    #     except: ht_df.loc[ ht_df[ht_df.season == 2023].index[i],'elo'] = team_ref_2023.loc[home_team_input,'Current ELO']






    # ############################################################### data preparation
    TEAM = Team(f'{away_team_input}')
    try:
        dfa = Team.Season(f'{away_team_input}').get_data(data=pbp)
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

    data_sum_off = ['game_id','air_yards','epa','passing_yards','rushing_yards','receiving_yards','interception','fumble_lost','touchdown','score_differential']
    data_sum_def = ['game_id','interception','fumble_lost','epa','passing_yards','rushing_yards','receiving_yards']

    off=dfa[dfa['posteam']==Team(f'{away_team_input}').team_abr]


    defense=dfa[dfa['defteam']==Team(f'{away_team_input}').team_abr]

    df_o = off[data_sum_off].groupby(by=['game_id']).sum()
    df_o['points_scored'] = 0
    df_d['points_allowed'] = 0

    df_d = defense[data_sum_def].groupby(by=['game_id']).sum()
    df_d = df_d.rename(columns = {'interception':'interceptions_gained','fumble_lost':'fumble_gained','epa':'epa_allowed','passing_yards':'passing_yards_allowed','rushing_yards':'rushing_yards_allowed','receiving_yards':'receiving_yards_allowed'} )


    for GID in df_o.index:

        df_o.loc[GID,'score_differential'] = off[off['game_id'] == GID].score_differential.iloc[len(off[off.index == GID])-1]
        if all(off.loc[off.game_id == GID,'home_team'] == Team(f'{away_team_input}').team_abr):
            df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
            df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
            df_o.loc[GID,'home_away'] = 1
        else:
            df_d.loc[GID,'points_allowed'] = max(off.loc[off.game_id == GID,'total_home_score'].dropna())
            df_o.loc[GID,'points_scored'] = max(off.loc[off.game_id == GID,'total_away_score'].dropna())
            df_o.loc[GID,'home_away'] = 0




    at_df = pd.concat([df_o,df_d],axis=1)

    at_df['season']=None

    for i in range(len(at_df)):
        if at_df.iloc[i].name[0:4] == '2022':
            at_df['season'].iloc[i] = 2022
        else: 
            at_df['season'].iloc[i] = 2023

    elo_temp_away_2022 = pd.Series(list(team_ref_2022.loc[away_team_input,'elo']))
    #elo_temp_away_2023 = pd.Series(list(team_ref_2023.loc[away_team_input,'elo']))

    at_df['elo']=None

    counter=0
    for i in range(len(at_df[at_df['season'] == 2022])):
        if i < (len(elo_temp_away_2022)-1):
            at_df.loc[ at_df[at_df.season == 2022].index[i],'elo'] = elo_temp_away_2022[i]
        else: 
            at_df.loc[ at_df[at_df.season == 2022].index[i],'elo'] = team_ref_2022.loc[away_team_input,'Current ELO'] + counter*20
            counter+=1



    # for i in range(len(at_df[at_df['season'] == 2023])):
    #     try:at_df.loc[ at_df[at_df.season == 2023].index[i],'elo'] = elo_temp_away_2023[i]
    #     except: at_df.loc[ at_df[at_df.season == 2023].index[i],'elo'] = team_ref_2023.loc[away_team_input,'Current ELO']




        
        
        
        
    at_df.elo = at_df.elo.astype('float')
    ht_df.elo = ht_df.elo.astype('float')  

    ht_df['turnovers'] = ht_df.fumble_lost + ht_df.interception
    at_df['turnovers'] = at_df.fumble_lost + at_df.interception
    ht_df['takeaways'] = ht_df.fumble_gained + ht_df.interceptions_gained
    at_df['takeaways'] = at_df.fumble_gained + at_df.interceptions_gained











    ################## home #####################

    predictors = ['epa', 'epa_allowed', 'passing_yards', 'rushing_yards', 'turnovers', 'takeaways','home_away', 'elo']
    y = ht_df['points_scored']
    weights = np.ones(len(ht_df))
    for i in range(len(weights)-1):
        weights[i+1] = weights[i] + .5 
        

    model_home = LinearRegression().fit(ht_df[predictors],y,sample_weight=weights)


    exp_epa = (stats.median(ht_df.epa) + stats.median(at_df.epa_allowed))/2
    exp_epa_allowed = (stats.median(ht_df.epa_allowed) + stats.median(at_df.epa))/2
    exp_passing_yards = (stats.median(ht_df.passing_yards) + stats.median(at_df.passing_yards_allowed))/2
    exp_rushing_yards = (stats.median(ht_df.rushing_yards) + stats.median(at_df.rushing_yards_allowed))/2
    exp_turnovers = (stats.mean(ht_df.turnovers) + stats.mean(at_df.takeaways))/2
    exp_takeaways = (stats.mean(ht_df.takeaways) + stats.mean(at_df.turnovers))/2

    home_score_proj = model_home.predict(np.array([[exp_epa,exp_epa_allowed,exp_passing_yards,exp_rushing_yards,exp_turnovers,exp_takeaways,1, team_ref_2022.loc[home_team_input,'Current ELO']]]))








    ################## away #####################


    predictors = ['epa', 'epa_allowed', 'passing_yards', 'rushing_yards', 'turnovers', 'takeaways','home_away', 'elo']
    y = at_df['points_scored']
    weights = np.ones(len(at_df))
    for i in range(len(weights)-1):
        weights[i+1] = weights[i] + .5
    model_away = LinearRegression().fit(at_df[predictors],y, sample_weight=weights)




    exp_epa = (stats.median(at_df.epa) + stats.median(ht_df.epa_allowed))/2
    exp_epa_allowed = (stats.median(at_df.epa_allowed) + stats.median(ht_df.epa))/2
    exp_passing_yards = (stats.median(at_df.passing_yards) + stats.median(ht_df.passing_yards_allowed))/2
    exp_rushing_yards = (stats.median(at_df.rushing_yards) + stats.median(ht_df.rushing_yards_allowed))/2
    exp_turnovers = (stats.mean(at_df.turnovers) + stats.mean(ht_df.takeaways))/2
    exp_takeaways = (stats.mean(at_df.takeaways) + stats.mean(ht_df.turnovers))/2



    away_score_proj = model_away.predict(np.array([[exp_epa,exp_epa_allowed,exp_passing_yards,exp_rushing_yards,exp_turnovers,exp_takeaways,0, team_ref_2022.loc[away_team_input,'Current ELO']]]))



    df.loc[s['games'][game]['id'],'Home_Team'] = f"{s['games'][game]['homeTeam']['fullName']}"
    df.loc[s['games'][game]['id'],'Away_Team'] = f"{s['games'][game]['awayTeam']['fullName']}"
    df.loc[s['games'][game]['id'],'Home_Score'] = home_score_proj
    df.loc[s['games'][game]['id'],'Away_Score'] = away_score_proj
    df.loc[s['games'][game]['id'],'Spread'] = home_score_proj - away_score_proj
    df.loc[s['games'][game]['id'],'Total'] = home_score_proj + away_score_proj



    print(f"{home_team_input}: {home_score_proj}")
    print(f"{away_team_input}: {away_score_proj}")







df









