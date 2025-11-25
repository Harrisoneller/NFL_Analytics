import nfl_data_py as nfl
import pandas as pd
import numpy as np
from random import choice, randint, uniform

# Set random seed for reproducibility
np.random.seed(42)
nfl.import_pbp_data([2024])  # Import play-by-play data for the 2024 season
# Function to fetch team stats from nfl_data_py
def get_team_stats(season, team):
    # Load play-by-play data for the specified season
    pbp = nfl.import_pbp_data([season])
    
    # Filter for the specified team on offense
    team_data = pbp[pbp['posteam'] == team]
    
    # Calculate basic stats
    pass_plays = team_data[team_data['play_type'] == 'pass']
    rush_plays = team_data[team_data['play_type'] == 'run']
    
    pass_yards_avg = pass_plays['yards_gained'].mean()
    rush_yards_avg = rush_plays['yards_gained'].mean()
    pass_prob = len(pass_plays) / (len(pass_plays) + len(rush_plays))
    pass_td_prob = pass_plays['pass_touchdown'].mean()
    rush_td_prob = rush_plays['rush_touchdown'].mean()
    
    return {
        'pass_yards_avg': pass_yards_avg,
        'rush_yards_avg': rush_yards_avg,
        'pass_prob': pass_prob,
        'pass_td_prob': pass_td_prob,
        'rush_td_prob': rush_td_prob
    }

# Function to simulate a single play
def simulate_play(team_stats, yard_line):
    play_type = 'pass' if uniform(0, 1) < team_stats['pass_prob'] else 'rush'
    yards_gained = 0
    touchdown = False
    
    if play_type == 'pass':
        yards_gained = int(np.random.normal(team_stats['pass_yards_avg'], 10))
        touchdown = uniform(0, 1) < team_stats['pass_td_prob']
    else:  # rush
        yards_gained = int(np.random.normal(team_stats['rush_yards_avg'], 5))
        touchdown = uniform(0, 1) < team_stats['rush_td_prob']
    
    # Cap yards gained at remaining distance to endzone
    yards_gained = min(yards_gained, 100 - yard_line)
    new_yard_line = yard_line + yards_gained
    
    return play_type, yards_gained, touchdown, new_yard_line

# Function to simulate a drive
def simulate_drive(team_stats, starting_yard_line):
    downs = 1
    yards_to_go = 10
    yard_line = starting_yard_line
    plays = []
    
    while downs <= 4:
        play_type, yards_gained, touchdown, new_yard_line = simulate_play(team_stats, yard_line)
        plays.append(f"Down {downs}: {play_type} for {yards_gained} yards")
        
        if touchdown:
            plays.append("Touchdown!")
            return plays, 7, yard_line  # 7 points for TD
        
        yard_line = new_yard_line
        yards_to_go -= yards_gained
        
        if yards_to_go <= 0:  # First down
            yards_to_go = 10
            downs = 1
        else:
            downs += 1
        
        if downs > 4:  # Turnover on downs
            plays.append("Turnover on downs")
            return plays, 0, yard_line
    
    return plays, 0, yard_line

# Function to simulate a full game
def simulate_game(team1, team2, season=2024):
    # Fetch team stats
    team1_stats = get_team_stats(season, team1)
    team2_stats = get_team_stats(season, team2)
    
    # Initialize game state
    team1_score = 0
    team2_score = 0
    team1_yards = 0
    team2_yards = 0
    current_team = team1
    yard_line = 25  # Start at own 25-yard line
    quarter = 1
    plays_log = []
    
    # Simulate 4 quarters (simplified: 6 drives per team)
    for _ in range(12):  # 6 drives each
        team_stats = team1_stats if current_team == team1 else team2_stats
        drive_plays, points, final_yard_line = simulate_drive(team_stats, yard_line)
        
        if current_team == team1:
            team1_score += points
            team1_yards += (final_yard_line - yard_line)
            plays_log.append(f"{team1} Drive: {' | '.join(drive_plays)}")
            current_team = team2
            yard_line = 25  # Opponent starts at their 25
        else:
            team2_score += points
            team2_yards += (final_yard_line - yard_line)
            plays_log.append(f"{team2} Drive: {' | '.join(drive_plays)}")
            current_team = team1
            yard_line = 25
        
        if _ == 5:  # Switch to Q3 after 6 drives
            quarter = 3
    
    # Game results
    winner = team1 if team1_score > team2_score else team2 if team2_score > team1_score else "Tie"
    return {
        'team1': team1,
        'team2': team2,
        'team1_score': team1_score,
        'team2_score': team2_score,
        'team1_yards': team1_yards,
        'team2_yards': team2_yards,
        'winner': winner,
        'plays_log': plays_log
    }

# Main execution
if __name__ == "__main__":
    # Simulate a game between two teams (e.g., Chiefs vs. Bills)
    team1 = "KC"  # Kansas City Chiefs
    team2 = "BUF"  # Buffalo Bills
    season = 2024
    
    result = simulate_game(team1, team2, season)
    
    # Print results
    print(f"Game Simulation: {result['team1']} vs {result['team2']}")
    print(f"Final Score: {result['team1']} {result['team1_score']} - {result['team2']} {result['team2_score']}")
    print(f"Total Yards: {result['team1']} {result['team1_yards']} - {result['team2']} {result['team2_yards']}")
    print(f"Winner: {result['winner']}")
    print("\nPlay-by-Play Log:")
    for play in result['plays_log']:
        print(play)