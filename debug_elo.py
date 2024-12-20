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
import scipy.stats as ss
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
# Calculate ELO rating for NFL teams using data scraped from web
# Starting out with calculating # of wins
from typing import Dict


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
    #team_ref = pd.DataFrame(season['winner'].append(season['loser']),columns=['team']).drop_duplicates().set_index(['team']).sort_index()
    team_ref = pd.DataFrame(pd.concat([season['winner'],season['loser']]),columns=['team']).drop_duplicates().set_index(['team']).sort_index()

    #initialize vars


    # Typed these values in from 538.com
    # teams in alphabetical order
    if input_season == 2022:

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
    else:
      elo_list = [
      [    1475], #Arizona Cardinals
      [    1500], #Atlanta Falcons
      [    1750], #Baltimore Ravens
      [    1725], #Buffalo Bills
      [    1315], #Carolina Panthers
      [    1439], #Chicago Bears
      [    1614], #Cincinatti Bengals
      [    1520], #Cleveland Browns
      [    1621], #Dallas Cowboys
      [    1481], #Denver Broncos
      [    1700], #Detroit Lions
      [    1560], #Green Bay 
      [    1580], #Houston Texans
      [    1500], #Indianapolis Colts
      [    1510], #Jax Jaguars
      [    1742], #KC Chiefs
      [    1415], #LV Raiders
      [    1440], #LA Chargers
      [    1516], #LA Rams
      [    1590], #Miami Dolphins
      [    1420], #Minn Vikings
      [    1370], #NE Patriots
      [    1440], #NO Saints
      [    1425], #NY Giants
      [    1510], #NY Jets
      [    1615], #Philly Eagles
      [    1450], #Pitt Steelers
      [    1720], #49ers
      [    1550], #Seahawks
      [    1500], #TB Buccaneers 
      [    1400], #Tenn Titans 
      [    1400]] #WSH Commanders


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



# out_elo, team_ref_2023= get_current_elo(input_season=2023)
# team_ref_2023['season']=None
# team_ref_2023['season']=2023
# flat_list = [item for sublist in elo_list for item in sublist]
# negated_list = [-item for item in flat_list]
# ss.rankdata(negated_list)

def nfl_token_gen():
    url = "https://api.nfl.com/v1/reroute"

    # TODO: resolve if DNT or x-domain-id are necessary.  pulled them from chrome inspector
    payload = "grant_type=client_credentials"
    headers = {"DNT": "1", "x-domain-id": "100", "Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.content)["access_token"]


def nfl_headers_gen():
    token = nfl_token_gen()
    return {
        "Host": "api.nfl.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nfl.com/",
        "authorization": f"Bearer {token}",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-User": "?1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

def nfl_game_schedule(season=2021, season_type="REG", week=1, headers=None, raw=False) -> Dict:
    """nfl_game_schedule()
    Args:
        season (int): season
        season_type (str): season type - REG, POST
        week (int): week
    Returns:
        Dict: Dictionary of odds and props data with keys
    Example:
        `nfl_df = nfl_game_schedule(
            season = 2021, seasonType='REG', week=1
        )`
    """
    if headers is None:
        headers = nfl_headers_gen()
    params = {"season": season, "seasonType": season_type, "week": week}
    pbp_txt = {}
    summary_url = "https://api.nfl.com/experience/v1/games"
    summary_resp = requests.get(summary_url, headers=headers, params=params)
    summary = summary_resp.json()

    incoming_keys_expected = [
        "id",
        "homeTeam",
        "awayTeam",
        "category",
        "date",
        "time",
        "broadcastInfo",
        "neutralSite",
        "venue",
        "season",
        "seasonType",
        "status",
        "week",
        "weekType",
        "externalIds",
        "ticketUrl",
        "ticketVendors",
        "detail",
    ]
    dict_keys_expected = ["homeTeam", "possessionTeam", "visitorTeam", "weather"]
    array_keys_expected = ["scoringSummaries", "drives", "plays"]
    return summary



nfl_game_schedule(season=2024, season_type="REG", week=1)