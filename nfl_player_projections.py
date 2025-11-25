# Comprehensive Python Script for NFL Offensive Player Projections
# Author: Grok 4 (synthesized from various open-source resources and methodologies)
# Date: November 21, 2025
# Purpose: Generate statistical projections for all offensive NFL players (QB, RB, WR, TE) using a hybrid approach:
#   - Historical baselines (averages and similarity scores)
#   - Advanced modeling (XGBoost for efficiency predictions)
#   - Simulations (Monte Carlo for game context, pace, injuries)
#   - Data from free sources via nflreadpy (play-by-play, weekly stats, rosters, injuries, schedules)
# Assumptions:
#   - User has Python 3.8+ with libraries: pip install nflreadpy polars pandas numpy xgboost scipy scikit-learn
#   - Projections for the upcoming week/season based on historical data up to the most recent completed week
#   - Current year: 2025 (adjust as needed)
#   - Outputs: CSV with per-player projections (e.g., passing_yards, rushing_yards, receptions, TDs, fantasy_points)
#   - Fantasy scoring: Standard PPR (adjustable)
#   - Handles ~500 offensive players league-wide
# Limitations: No real-time data fetching in script; assumes data is up-to-date via nflreadpy cache
#              Injury simulation is probabilistic based on historical rates
# Usage: Run the script; it will generate 'player_projections.csv'

import nflreadpy as nfl  # Replacement for nfl_data_py
import polars as pl  # nflreadpy returns Polars DataFrames
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import poisson, norm  # For simulations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuration
START_YEAR = 2020  # Updated to recent years to avoid server issues with older data
CURRENT_YEAR = 2025
POSITIONS = ['QB', 'RB', 'WR', 'TE']  # Offensive skill positions
SIM_ITERATIONS = 1000  # Number of Monte Carlo simulations per player/game
PPR_SCORING = True  # True for PPR, False for standard
INJURY_RATE = 0.05  # Base injury probability (adjust based on player history)
UPCOMING_WEEK = 12  # Example: Project for week 12; set to current +1
FEATURES = ['age', 'position', 'team_pace', 'target_share', 'opp_def_rank', 'epa_per_pass', 'injury_prob']

# Step 1: Data Ingestion (using nflreadpy for free historical data)
def load_nfl_data():
    print("Loading NFL data...")
    years = list(range(START_YEAR, CURRENT_YEAR + 1))
    years_past = list(range(START_YEAR, CURRENT_YEAR))  # Exclude 2025 for injuries since data not available
    ngs_years = [y for y in years if y >= 2016]  # Next Gen Stats available from 2016
    # Rosters: Player info, positions, teams
    rosters = nfl.load_rosters(seasons=years).to_pandas()
    rosters = rosters[rosters['position'].isin(POSITIONS)]
    # Standardize IDs and names in rosters
    if 'gsis_id' in rosters.columns:
        rosters = rosters.rename(columns={'gsis_id': 'player_id'})
    if 'full_name' in rosters.columns:
        rosters = rosters.rename(columns={'full_name': 'player_name'})
    elif 'first_name' in rosters.columns and 'last_name' in rosters.columns:
        rosters['player_name'] = rosters['first_name'] + ' ' + rosters['last_name']
    
    # Play-by-Play: Detailed plays for advanced metrics (EPA, etc.)
    pbp = nfl.load_pbp(seasons=years).to_pandas()
    
    # Weekly Player Stats: Aggregated per week (yards, TDs, etc.)
    weekly = nfl.load_player_stats(seasons=years).to_pandas()
    weekly = weekly[weekly['position'].isin(POSITIONS)]
    
    # Injuries: Historical injury data for probability estimation
    injuries = nfl.load_injuries(seasons=years_past).to_pandas()
    
    # Schedules: For matchups, home/away
    schedules = nfl.load_schedules(seasons=years).to_pandas()
    
    # Next Gen Stats (if available, else approximate)
    ngs = nfl.load_nextgen_stats(stat_type='passing', seasons=ngs_years).to_pandas()  # Example for passing
    
    return rosters, pbp, weekly, injuries, schedules, ngs

# Step 2: Preprocessing and Feature Engineering
def preprocess_data(rosters, pbp, weekly, injuries, schedules, ngs):
    print("Preprocessing data...")
    
    # Standardize player_id (assuming gsis_id in rosters, player_id in weekly)
    if 'gsis_id' in rosters.columns:
        rosters = rosters.rename(columns={'gsis_id': 'player_id'})
    if 'full_name' in rosters.columns:
        rosters = rosters.rename(columns={'full_name': 'player_name'})
    elif 'first_name' in rosters.columns and 'last_name' in rosters.columns:
        rosters['player_name'] = rosters['first_name'] + ' ' + rosters['last_name']
    
    # Calculate age from birth_date
    if 'birth_date' in rosters.columns:
        rosters['birth_date'] = pd.to_datetime(rosters['birth_date'], errors='coerce')
        rosters['age'] = CURRENT_YEAR - rosters['birth_date'].dt.year
        rosters['age'] = rosters['age'].fillna(rosters['age'].median())  # Fill NaNs with median age
    
    # Merge rosters with weekly stats
    merge_cols = ['player_id', 'player_name', 'age', 'depth_chart_position']  # Removed 'team' to avoid conflict
    available_cols = [col for col in merge_cols if col in rosters.columns]
    data = pd.merge(weekly, rosters[available_cols], on='player_id', how='left')
    
    # Fallback for player_name if not present
    if 'player_name' not in data.columns:
        data['player_name'] = data.get('player_display_name', data['player_id'])
    
    # Calculate advanced metrics from PBP
    # Example: EPA per play for QBs
    qb_pbp = pbp[pbp['posteam_type'] == 'pass']
    epa_mean = qb_pbp.groupby('passer_player_id')['epa'].mean()
    data['epa_per_pass'] = data['player_id'].map(epa_mean).fillna(0)
    
    # Opponent-adjusted metrics (e.g., defense strength)
    def_rank = pbp.groupby('defteam')['epa'].mean().rank()
    data['opp_def_rank'] = data['opponent_team'].map(def_rank)
    
    # Usage shares (targets, carries)
    team_targets = data.groupby(['season', 'week', 'team'])['targets'].sum()
    data['target_share'] = data.apply(lambda row: row['targets'] / team_targets.get((row['season'], row['week'], row['team']), 1), axis=1)
    
    # Injury history: Approx prob based on missed games
    missed_games = injuries[injuries['report_status'] == 'Out'].groupby('gsis_id').size()
    player_seasons = data.groupby('player_id')['season'].nunique()
    data['injury_prob'] = data['player_id'].map(missed_games).fillna(0) / (data['player_id'].map(player_seasons) * 17)
    data['injury_prob'] = data['injury_prob'].clip(0, 1).fillna(INJURY_RATE)
    
    # Pace and plays: From schedules and PBP
    team_pace = pbp.groupby('posteam').apply(lambda g: g['play_id'].count() / g['game_id'].nunique())
    data['team_pace'] = data['team'].map(team_pace)
    
    # Filter to completed weeks
    data = data[data['week'] < UPCOMING_WEEK]
    
    # Label encode categoricals
    le_pos = LabelEncoder()
    data['position'] = le_pos.fit_transform(data['position'])
    le_opp = LabelEncoder()
    data['opponent_team'] = le_opp.fit_transform(data['opponent_team'].astype(str))
    
    return data, le_pos

# Step 3: Historical Baselines (Similarity Scores and Averages)
def historical_baselines(data):
    print("Computing historical baselines...")
    
    # Per-player season averages
    hist_avg = data.groupby(['player_id', 'season'])[['completions', 'attempts', 'passing_yards', 'passing_tds',
                                                      'passing_interceptions', 'carries', 'rushing_yards', 'rushing_tds',
                                                      'receptions', 'targets', 'receiving_yards', 'receiving_tds']].mean().reset_index()
    
    # Similarity scores: Euclidean distance on key stats (for comparable players)
    def similarity_score(player_stats, hist_data):
        from sklearn.metrics import pairwise_distances
        key_cols = ['age', 'passing_yards', 'rushing_yards', 'receiving_yards']
        if set(key_cols).issubset(hist_data.columns):
            dist = pairwise_distances(player_stats[key_cols].values.reshape(1, -1), hist_data[key_cols].values)
            similar_idx = np.argmin(dist)
            return hist_data.iloc[similar_idx]
        else:
            return pd.Series()  # Fallback
    
    # Example: For each player, find historical twin
    baselines = {}
    for pid in data['player_id'].unique():
        p_data = data[data['player_id'] == pid]
        if p_data.empty:
            continue
        p_stats = p_data.iloc[-1]  # Last available
        player_hist = hist_avg[hist_avg['player_id'] == pid]
        if not player_hist.empty:
            twin = similarity_score(p_stats, player_hist)
            baselines[pid] = twin
    
    return pd.DataFrame(baselines).T

# Step 4: Advanced Modeling (XGBoost for Stat Predictions)
def train_models(data, le_pos):
    print("Training advanced models...")
    
    targets = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 
               'completions', 'attempts', 'passing_interceptions', 'carries', 'receptions', 'targets']
    
    models = {}
    for pos in POSITIONS:
        pos_encoded = le_pos.transform([pos])[0]
        pos_data = data[data['position'] == pos_encoded]
        X = pos_data[FEATURES]
        for target in targets:
            y = pos_data[target]
            if y.empty or y.sum() == 0:  # Skip if no data
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))
            print(f"{pos} {target} MAE: {mae}")
            models[(pos, target)] = model
    
    return models

# Step 5: Simulations (Monte Carlo for Game Context)
def run_simulations(data, models, schedules, le_pos, active_players):
    print("Running simulations...")
    
    projections = []
    upcoming_games = schedules[schedules['week'] == UPCOMING_WEEK]
    
    for _, game in upcoming_games.iterrows():
        home_team = game.get('home_team', game.get('home_team_code'))  # Handle possible column variations
        away_team = game.get('away_team', game.get('away_team_code'))
        # Simulate team-level: plays, pace
        home_pace = data[data['team'] == home_team]['team_pace'].mean()
        sim_plays = norm.rvs(loc=home_pace, scale=5, size=SIM_ITERATIONS)  # Normal dist for plays
        
        for team in [home_team, away_team]:
            team_active = active_players[active_players['team'] == team]
            for _, active_player in team_active.iterrows():
                pid = active_player['player_id']
                player_data = data[data['player_id'] == pid].sort_values(['season', 'week'], ascending=False).head(1)
                if player_data.empty:
                    continue
                player = player_data.iloc[0]
                pos_encoded = player['position']
                pos = le_pos.inverse_transform([pos_encoded])[0]
                sim_stats = {
                    'passing_yards': [], 'passing_tds': [], 'completions': [], 'attempts': [], 'passing_interceptions': [],
                    'rushing_yards': [], 'rushing_tds': [], 'carries': [],
                    'receiving_yards': [], 'receiving_tds': [], 'receptions': [], 'targets': []
                }
                
                for i in range(SIM_ITERATIONS):
                    # Injury check
                    if np.random.rand() < player['injury_prob']:
                        for stat in sim_stats:
                            sim_stats[stat].append(0)
                        continue
                    
                    # Model baseline for each target
                    input_feats = player[FEATURES].values.reshape(1, -1)
                    for target in sim_stats.keys():
                        if (pos, target) in models:
                            base_val = max(0, models[(pos, target)].predict(input_feats)[0])
                        else:
                            base_val = 0
                        
                        # Simulate variability: Poisson for counts (TDs, completions, etc.), Normal for yards
                        if 'yards' in target:
                            sim_val = norm.rvs(loc=base_val, scale=50) * (sim_plays[i] / home_pace if home_pace else 1)  # Adjust for pace
                        else:
                            sim_val = poisson.rvs(mu=base_val)
                        
                        sim_stats[target].append(max(0, sim_val))  # Non-negative
                        
                    if i % 100 == 0:
                        print(i)
                print(f"Player {player['player_name']} sim {i} complete")
                
                # Aggregate sim means
                proj = {stat: np.mean(vals) for stat, vals in sim_stats.items()}
                proj['player_id'] = pid
                proj['player_name'] = player.get('player_name', active_player.get('player_name', pid))
                proj['position'] = pos
                proj['team'] = team
                proj['fantasy_points'] = calculate_fantasy_points(proj, PPR_SCORING)
                projections.append(proj)
    
    return pd.DataFrame(projections)

# Helper: Calculate fantasy points
def calculate_fantasy_points(stats, ppr=True):
    points = (stats.get('passing_yards', 0) / 25) + (stats.get('passing_tds', 0) * 4) - stats.get('passing_interceptions', 0) * 2
    points += (stats.get('rushing_yards', 0) / 10) + (stats.get('rushing_tds', 0) * 6)
    points += (stats.get('receiving_yards', 0) / 10) + (stats.get('receiving_tds', 0) * 6)
    if ppr:
        points += stats.get('receptions', 0)
    return points

# Step 6: Combine Methodologies and Output
def generate_projections():
    rosters, pbp, weekly, injuries, schedules, ngs = load_nfl_data()
    data, le_pos = preprocess_data(rosters, pbp, weekly, injuries, schedules, ngs)
    active_players = rosters[rosters['season'] == CURRENT_YEAR]
    baselines = historical_baselines(data)
    models = train_models(data, le_pos)
    sim_projs = run_simulations(data, models, schedules, le_pos, active_players)
    
    # Ensemble: Average baselines, model preds, sim means (simple merge for demo)
    # For full ensemble, weight and average; here, use sims as primary
    final_projs = sim_projs.merge(baselines, on='player_id', suffixes=('_sim', '_hist'), how='left')
    # Could add model direct preds, but sims already incorporate
    
    final_projs.to_csv('player_projections.csv', index=False)
    print("Projections saved to player_projections.csv")

if __name__ == "__main__":
    generate_projections()