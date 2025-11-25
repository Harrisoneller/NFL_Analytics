import sportsdataverse.nfl as nfl
import pandas as pd
import numpy as np
from random import uniform
from scipy.stats import norm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Function to fetch and weight historical PBP data
def fetch_weighted_pbp(seasons=[2023, 2024, 2025], recency_weight=0.1):
    from datetime import datetime
    current_year = datetime.now().year
    seasons = [s for s in seasons if s <= current_year]  # Filter future seasons
    pbp = pd.concat([nfl.load_nfl_pbp(seasons=[s], return_as_pandas=True) for s in seasons])
    pbp['game_date'] = pd.to_datetime(pbp['game_date'])
    max_date = pbp['game_date'].max()
    pbp['recency_weight'] = np.exp(-recency_weight * (max_date - pbp['game_date']).dt.days)
    
    # Compute league averages as fallback
    league_epa = pbp['epa'].mean()
    league_success = (pbp['epa'] > 0).mean()
    league_pass_ypa = pbp[pbp['play_type'] == 'pass']['yards_gained'].mean()
    league_rush_ypa = pbp[pbp['play_type'] == 'run']['yards_gained'].mean()
    league_pass_prob = len(pbp[pbp['play_type'] == 'pass']) / len(pbp) if len(pbp) > 0 else 0.5
    league_pass_td = pbp[pbp['play_type'] == 'pass']['pass_touchdown'].mean()
    league_rush_td = pbp[pbp['play_type'] == 'run']['rush_touchdown'].mean()
    
    league_avg = {
        'epa_per_play': league_epa,
        'success_rate': league_success,
        'pass_ypa': league_pass_ypa,
        'rush_ypa': league_rush_ypa,
        'pass_prob': league_pass_prob,
        'pass_td_rate': league_pass_td,
        'rush_td_rate': league_rush_td,
        'weighted_ypp': np.average(pbp['yards_gained'].fillna(0), weights=pbp['recency_weight']),
        'proe': 0,
        'yards_per_drive': pbp.groupby('drive')['yards_gained'].sum().mean(),
        'yards_per_drive_allowed': pbp.groupby('drive')['yards_gained'].sum().mean()  # Same for league
    }
    
    # Precompute stats for all teams
    teams = pbp['posteam'].unique()
    team_stats = {'off': {}, 'def': {}}
    team_drives = {}
    
    for team in teams:
        team_stats['off'][team] = get_advanced_team_stats(pbp, team, is_offense=True, league_avg=league_avg)
        team_stats['def'][team] = get_advanced_team_stats(pbp, team, is_offense=False, league_avg=league_avg)
        team_stats['def'][team]['proe'] = 0  # PROE is offensive-only
        
        # Compute avg drives per game
        team_games = pbp[pbp['posteam'] == team]['game_id'].unique()
        total_drives = 0
        total_yards = 0
        for gid in team_games:
            game_data = pbp[(pbp['game_id'] == gid) & (pbp['posteam'] == team)]
            drives = game_data['drive'].unique()
            total_drives += len(drives)
            for d in drives:
                drive_data = game_data[game_data['drive'] == d]
                total_yards += drive_data['yards_gained'].sum()
        team_drives[team] = total_drives / len(team_games) if team_games.size > 0 else 11
        avg_yards_per_drive = total_yards / total_drives if total_drives > 0 else 30  # Default ~30 yds/drive
        team_stats['off'][team]['yards_per_drive'] = avg_yards_per_drive
        
        # Compute for def (allowed)
        def_games = pbp[pbp['defteam'] == team]['game_id'].unique()
        total_def_drives = 0
        total_allowed_yards = 0
        for gid in def_games:
            game_data = pbp[(pbp['game_id'] == gid) & (pbp['defteam'] == team)]
            drives = game_data['drive'].unique()
            total_def_drives += len(drives)
            for d in drives:
                drive_data = game_data[game_data['drive'] == d]
                total_allowed_yards += drive_data['yards_gained'].sum()
        avg_allowed_per_drive = total_allowed_yards / total_def_drives if total_def_drives > 0 else 30
        team_stats['def'][team]['yards_per_drive_allowed'] = avg_allowed_per_drive
        
        # Compute Pass Rate Over Expected (PROE)
        team_data = pbp[pbp['posteam'] == team]
        expected_pass_rate = 0.6  # League avg, or compute dynamically
        actual_pass_rate = team_stats['off'][team]['pass_prob']
        proe = actual_pass_rate - expected_pass_rate
        team_stats['off'][team]['proe'] = proe
    
    return pbp, league_avg, team_stats, team_drives

# Function to compute advanced team stats (weighted, opponent-adjusted)
def get_advanced_team_stats(pbp, team, is_offense=True, league_avg=None):
    if is_offense:
        data = pbp[pbp['posteam'] == team].copy()
        opp_col = 'defteam'
    else:
        data = pbp[pbp['defteam'] == team].copy()
        opp_col = 'posteam'
    
    if data.empty or data['recency_weight'].sum() == 0:
        return league_avg  # Fallback to league averages
    
    # Handle missing values
    data['epa'] = data['epa'].fillna(0)
    
    # Compute metrics
    data['success'] = (data['epa'] > 0).astype(int)
    
    weighted_stats = {
        'epa_per_play': np.average(data['epa'], weights=data['recency_weight']),
        'success_rate': np.average(data['success'], weights=data['recency_weight']),
        'pass_ypa': np.average(data[data['play_type'] == 'pass']['yards_gained'].fillna(0), weights=data[data['play_type'] == 'pass']['recency_weight']) if not data[data['play_type'] == 'pass'].empty else league_avg['pass_ypa'],
        'rush_ypa': np.average(data[data['play_type'] == 'run']['yards_gained'].fillna(0), weights=data[data['play_type'] == 'run']['recency_weight']) if not data[data['play_type'] == 'run'].empty else league_avg['rush_ypa'],
        'pass_prob': len(data[data['play_type'] == 'pass']) / len(data) if len(data) > 0 else league_avg['pass_prob'],
        'pass_td_rate': data[data['play_type'] == 'pass']['pass_touchdown'].fillna(0).mean(),
        'rush_td_rate': data[data['play_type'] == 'run']['rush_touchdown'].fillna(0).mean(),
        'weighted_ypp': np.average(data['yards_gained'].fillna(0), weights=data['recency_weight'])
    }
    
    # Opponent adjustment (average opponent strength)
    opp_strength = data.groupby(opp_col)['epa'].mean().mean()  # Simple avg opp EPA
    for key in ['epa_per_play', 'success_rate', 'pass_ypa', 'rush_ypa']:
        weighted_stats[key] -= 0.1 * opp_strength  # Adjust down if facing strong opps
    
    return weighted_stats

# Function to prepare adjusted stats for a matchup
def get_matchup_adjusted_stats(pbp, team1, team2, league_avg):
    team1_off = get_advanced_team_stats(pbp, team1, is_offense=True, league_avg=league_avg)
    team1_def = get_advanced_team_stats(pbp, team1, is_offense=False, league_avg=league_avg)
    team2_off = get_advanced_team_stats(pbp, team2, is_offense=True, league_avg=league_avg)
    team2_def = get_advanced_team_stats(pbp, team2, is_offense=False, league_avg=league_avg)
    
    # Adjust team1 off vs team2 def
    adj_stats_t1 = {k: (team1_off[k] + team2_def[k]) / 2 for k in team1_off}
    adj_stats_t2 = {k: (team2_off[k] + team1_def[k]) / 2 for k in team2_off}
    
    return adj_stats_t1, adj_stats_t2

# ML Model for baseline predictions
def train_ml_model(pbp, team_stats, league_avg):
    # Prepare features: team stats, opponent stats, home/away
    features = []
    targets = []
    
    for game_id in pbp['game_id'].unique():
        game = pbp[pbp['game_id'] == game_id]
        home = game['home_team'].iloc[0]
        away = game['away_team'].iloc[0]
        
        past_pbp = pbp[pbp['game_date'] < game['game_date'].min()]
        
        home_off = team_stats['off'].get(home, league_avg)
        away_def = team_stats['def'].get(away, league_avg)
        
        feat = list(home_off.values()) + list(away_def.values()) + [1]  # Home indicator
        features.append(feat)
        targets.append(game[game['posteam'] == home]['total_home_score'].max())  # Home score
    
    X = np.array(features)
    y = np.array(targets)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test))}")
    return model

# Simulate a single play (enhanced with advanced metrics)
def simulate_play(adj_stats, yard_line, down, to_go):
    # Situational pass prob: increase on 3rd/4th and long
    base_pass_prob = adj_stats['pass_prob'] + adj_stats.get('proe', 0)
    if down >= 3 and to_go > 5:
        pass_prob = min(0.9, base_pass_prob + 0.2)
    else:
        pass_prob = base_pass_prob
    
    play_type = 'pass' if uniform(0, 1) < pass_prob else 'rush'
    mean_yards = adj_stats['weighted_ypp'] * (1 + adj_stats['epa_per_play'] / 2) * 1.2  # Boost mean
    std_yards = mean_yards * (1 - adj_stats['success_rate']) * 2.5  # Increase variance
    
    # Use lognormal for explosive plays
    yards_gained = max(-5, min(80, int(np.random.lognormal(np.log(mean_yards), std_yards / mean_yards))))
    yards_gained = min(yards_gained, 100 - yard_line)
    
    td_rate = (adj_stats['pass_td_rate'] if play_type == 'pass' else adj_stats['rush_td_rate']) * 1.5  # Boost
    if yard_line > 80:  # Red zone boost
        td_rate *= 1.5
    
    touchdown = (yards_gained >= 100 - yard_line) or (uniform(0, 1) < td_rate * (1 - yard_line / 100) * (1 + adj_stats['epa_per_play']))
    
    epa = adj_stats['epa_per_play'] + uniform(-1, 1) * (1 - adj_stats['success_rate'])
    
    # Add rare turnover
    if uniform(0, 1) < 0.03:  # ~3% chance
        yards_gained = 0
        epa -= 3  # Turnover EPA penalty
    
    return {'play_type': play_type, 'yards_gained': yards_gained, 'touchdown': touchdown, 'epa': epa}

# Simulate a drive (more realistic downs management)
def simulate_drive(adj_stats, starting_yard_line=25):
    yard_line = starting_yard_line
    down = 1
    to_go = 10
    total_yards = 0
    total_epa = 0
    points = 0
    play_count = 0
    
    while down <= 4 and yard_line < 100 and play_count < 20:  # Increase max plays
        play = simulate_play(adj_stats, yard_line, down, to_go)
        yard_line += play['yards_gained']
        total_yards += play['yards_gained']
        total_epa += play['epa']
        play_count += 1
        
        # Add random penalty (small chance)
        if uniform(0, 1) < 0.1:
            penalty_yards = np.random.choice([-5, 5, 10, -10], p=[0.3, 0.3, 0.2, 0.2])
            yard_line += penalty_yards
            total_yards += penalty_yards
        
        if play['touchdown']:
            points = 7
            if uniform(0, 1) < 0.95:  # XP
                points += 1
            elif uniform(0, 1) < 0.2:  # 2-pt attempt, 50% success
                if uniform(0, 1) < 0.5:
                    points += 2
            break
        
        to_go -= play['yards_gained']
        if to_go <= 0 or uniform(0, 1) < 0.1:  # Small chance to force first down for longer drives
            down = 1
            to_go = 10
        else:
            down += 1
        
        # Ensure min plays
        if play_count < 4 and down > 4:
            down = 3  # Extend slightly
    
    # Field goal or punt logic
    if down > 4:
        if yard_line >= 60 and uniform(0, 1) < 0.8:  # 80% attempt FG if in range
            fg_dist = 117 - yard_line
            fg_prob = max(0.4, 1 - (fg_dist - 17) / 40)
            if uniform(0, 1) < fg_prob:
                points = 3
        else:
            # Punt: advance field ~40 yards (simplified turnover)
            pass
    
    # EPA-based calibration
    if adj_stats['epa_per_play'] > 0.1:
        points += np.random.choice([0, 3], p=[0.8, 0.2])  # Small chance extra FG
    
    return total_yards, points, total_epa

# Simulate full game
def simulate_game(adj_stats_t1, adj_stats_t2, team_drives, team1, team2):
    # Calculate num_drives: average of teams' historical, adjusted by EPA
    avg_drives_t1 = team_drives.get(team1, 11)
    avg_drives_t2 = team_drives.get(team2, 11)
    pace_adjust = 1 if adj_stats_t1['epa_per_play'] > 0.1 else -1  # Simple adjustment
    num_drives = round((avg_drives_t1 + avg_drives_t2) / 2) + pace_adjust
    
    t1_score, t2_score = 0, 0
    t1_yards, t2_yards = 0, 0
    t1_epa, t2_epa = 0, 0
    
    for _ in range(num_drives // 2):
        y, p, e = simulate_drive(adj_stats_t1)
        t1_yards += y
        t1_score += p
        t1_epa += e
        
        y, p, e = simulate_drive(adj_stats_t2)
        t2_yards += y
        t2_score += p
        t2_epa += e
    
    # Overtime sim (simple)
    if t1_score == t2_score and uniform(0,1) < 0.1:
        y, p, _ = simulate_drive(adj_stats_t1)
        t1_score += p
        t1_yards += y
    
    return t1_score, t2_score, t1_yards, t2_yards, t1_epa, t2_epa

# Monte Carlo simulation with ML ensemble
def project_game(team1, team2, pbp=None, league_avg=None, team_stats=None, team_drives=None, ml_model=None, num_sims=2000):
    if pbp is None:
        pbp, league_avg, team_stats, team_drives = fetch_weighted_pbp()
    if ml_model is None:
        ml_model = train_ml_model(pbp, team_stats, league_avg)
    
    adj_t1 = team_stats['off'].get(team1, league_avg)
    adj_t2 = team_stats['off'].get(team2, league_avg)
    def_t2 = team_stats['def'].get(team2, league_avg)
    def_t1 = team_stats['def'].get(team1, league_avg)
    
    # Adjust shared keys
    shared_keys = [k for k in adj_t1 if k not in ['proe', 'yards_per_drive']]
    for k in shared_keys:
        adj_t1[k] = (adj_t1[k] + def_t2[k]) / 2
        adj_t2[k] = (adj_t2[k] + def_t1[k]) / 2
    
    # Adjust offense-specific
    adj_t1['yards_per_drive'] = (adj_t1['yards_per_drive'] + def_t2['yards_per_drive_allowed']) / 2
    adj_t2['yards_per_drive'] = (adj_t2['yards_per_drive'] + def_t1['yards_per_drive_allowed']) / 2
    
    # ML baseline prediction
    feat_t1 = list(adj_t1.values()) + list(adj_t2.values()) + [1]  # Home for team1
    ml_score_t1 = ml_model.predict(np.array([feat_t1]))[0]
    feat_t2 = list(adj_t2.values()) + list(adj_t1.values()) + [0]
    ml_score_t2 = ml_model.predict(np.array([feat_t2]))[0]
    
    # Baseline yards: drives * ypp
    ml_yards_t1 = team_drives.get(team1, 11) * adj_t1['weighted_ypp']
    ml_yards_t2 = team_drives.get(team2, 11) * adj_t2['weighted_ypp']
    
    # Simulations
    t1_scores, t2_scores, t1_yards_list, t2_yards_list = [], [], [], []
    t1_wins = 0
    
    for _ in range(num_sims):
        t1_s, t2_s, t1_y, t2_y, _, _ = simulate_game(adj_t1, adj_t2, team_drives, team1, team2)
        t1_scores.append(t1_s)
        t2_scores.append(t2_s)
        t1_yards_list.append(t1_y)
        t2_yards_list.append(t2_y)
        
        if t1_s > t2_s:
            t1_wins += 1
        elif t1_s == t2_s:
            t1_wins += 0.5
    
    avg_t1_score = np.mean(t1_scores)
    avg_t2_score = np.mean(t2_scores)
    avg_t1_yards = np.mean(t1_yards_list)
    avg_t2_yards = np.mean(t2_yards_list)
    
    # Scale if too low
    hist_avg_yards_t1 = team_stats['off'].get(team1, league_avg)['yards_per_drive'] * team_drives.get(team1, 11)
    if avg_t1_yards < hist_avg_yards_t1 * 0.8:
        avg_t1_yards *= 1.2
    
    hist_avg_yards_t2 = team_stats['off'].get(team2, league_avg)['yards_per_drive'] * team_drives.get(team2, 11)
    if avg_t2_yards < hist_avg_yards_t2 * 0.8:
        avg_t2_yards *= 1.2
    
    # Ensemble: Average ML and sim
    final_t1_score = (ml_score_t1 + avg_t1_score) / 2
    final_t2_score = (ml_score_t2 + avg_t2_score) / 2
    final_t1_yards = (ml_yards_t1 + avg_t1_yards) / 2
    final_t2_yards = (ml_yards_t2 + avg_t2_yards) / 2
    win_prob_t1 = t1_wins / num_sims
    
    return {
        'projected_score': (final_t1_score, final_t2_score),
        'projected_yards': (final_t1_yards, final_t2_yards),
        'win_prob': (win_prob_t1, 1 - win_prob_t1),
        'ml_baseline': (ml_score_t1, ml_score_t2)
    }

# Example usage
if __name__ == "__main__":
    pbp, league_avg, team_stats, team_drives = fetch_weighted_pbp()
    model = train_ml_model(pbp, team_stats, league_avg)
    results = project_game("KC", "BUF", pbp, league_avg, team_stats, team_drives, model)
    print(f"Projected Score: KC {results['projected_score'][0]:.1f} - BUF {results['projected_score'][1]:.1f}")
    print(f"Projected Yards: KC {results['projected_yards'][0]:.0f} - BUF {results['projected_yards'][1]:.0f}")
    print(f"Win Probability: KC {results['win_prob'][0]*100:.1f}% - BUF {results['win_prob'][1]*100:.1f}%")
