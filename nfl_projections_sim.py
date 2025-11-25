import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import nflreadpy as nfl
from scipy.stats import norm, multivariate_normal  # For simulation

def weighted_cov(X, w):
    X = np.asarray(X)
    w = np.asarray(w)
    m = np.average(X, axis=0, weights=w)
    X_centered = X - m
    cov = np.dot((X_centered.T * w), X_centered) / w.sum()
    return cov

def compute_game_level_stats(pbp, schedules):
    # Filter to offensive plays
    pbp_off = pbp[(pbp['pass_attempt'] == 1) | (pbp['rush_attempt'] == 1)]

    # Aggregate offensive stats per game per team
    game_off = pbp_off.groupby(['game_id', 'posteam']).agg({
        'yards_gained': 'sum',
        'pass_attempt': 'sum',
        'rush_attempt': 'sum',
        'epa': 'sum',
        'play_id': 'count',  # Use 'play_id' for count of plays
        'season': 'first',
        'game_date': 'first',
        'sack': 'sum',  # Sacks allowed (offense perspective)
        'interception': 'sum',
        'fumble_lost': 'sum',
        'success': 'mean'  # Success rate
    }).reset_index()
    game_off.rename(columns={
        'posteam': 'offense',
        'yards_gained': 'total_yards',
        'play_id': 'total_plays',
        'epa': 'total_epa',
        'sack': 'sacks_allowed',
        'success': 'success_rate'
    }, inplace=True)
    game_off['turnovers'] = game_off['interception'] + game_off['fumble_lost']

    # Passing specific
    passing = pbp_off[pbp_off['pass_attempt'] == 1].groupby(['game_id', 'posteam']).agg({
        'yards_gained': 'sum',
        'epa': 'sum',
        'season': 'first',
        'game_date': 'first'
    }).reset_index().rename(columns={
        'posteam': 'offense',
        'yards_gained': 'passing_yards',
        'epa': 'pass_epa'
    })

    # Rushing specific
    rushing = pbp_off[pbp_off['rush_attempt'] == 1].groupby(['game_id', 'posteam']).agg({
        'yards_gained': 'sum',
        'epa': 'sum',
        'season': 'first',
        'game_date': 'first'
    }).reset_index().rename(columns={
        'posteam': 'offense',
        'yards_gained': 'rushing_yards',
        'epa': 'rush_epa'
    })

    # Merge
    game_off = game_off.merge(passing, on=['game_id', 'offense'], how='left')
    game_off = game_off.merge(rushing, on=['game_id', 'offense'], how='left')

    # Add defense, hfa
    games = pbp[['game_id', 'home_team', 'away_team', 'location']].drop_duplicates()
    game_off = game_off.merge(games, on='game_id')
    game_off['defense'] = np.where(game_off['offense'] == game_off['home_team'], game_off['away_team'], game_off['home_team'])
    game_off['hfa'] = np.where(game_off['offense'] == game_off['home_team'], 1, -1)
    game_off.loc[game_off['location'] == 'Neutral', 'hfa'] = 0

    # Add weather from schedules
    weather_cols = schedules[['game_id', 'roof', 'temp', 'wind']].drop_duplicates()
    game_off = game_off.merge(weather_cols, on='game_id', how='left')

    return game_off

def get_adjusted_ratings(df, stat, alpha_values=[50, 100, 150, 200, 250, 300, 350, 400]):
    df_stat = df[['offense', 'hfa', 'defense', stat, 'game_date', 'roof', 'temp', 'wind']].dropna().copy()
    df_stat['game_date'] = pd.to_datetime(df_stat['game_date'])
    max_date = df_stat['game_date'].max()
    days_ago = (max_date - df_stat['game_date']).dt.days
    decay_rate = np.log(2) / 365  # Half-life of 1 year; adjust as needed
    weights = np.exp(-decay_rate * days_ago)

    # Handle weather
    df_stat['temp'] = df_stat['temp'].fillna(72)  # Assume indoor temp
    df_stat['wind'] = df_stat['wind'].fillna(0)   # No wind indoors

    dummies = pd.get_dummies(df_stat[['offense', 'hfa', 'defense', 'roof']])
    dummies = pd.concat([dummies, df_stat[['temp', 'wind']]], axis=1)

    # XGBoost with grid search for reg_alpha
    param_grid = {'reg_alpha': alpha_values}
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(dummies, df_stat[stat], sample_weight=weights)
    best_model = grid_search.best_estimator_

    # Approximate ratings using feature importances (for continuity with original)
    importances = pd.DataFrame({'coef_name': dummies.columns, 'importance': best_model.feature_importances_})
    off_df = importances[importances['coef_name'].str.startswith('offense_')].copy()
    off_df['team'] = off_df['coef_name'].str.replace('offense_', '')
    off_df = off_df[['team', 'importance']].rename(columns={'importance': f'adj_off_{stat}'})
    def_df = importances[importances['coef_name'].str.startswith('defense_')].copy()
    def_df['team'] = def_df['coef_name'].str.replace('defense_', '')
    def_df = def_df[['team', 'importance']].rename(columns={'importance': f'adj_def_{stat}'})
    hfa_val = importances.loc[importances['coef_name'] == 'hfa', 'importance'].values[0] if 'hfa' in importances['coef_name'].values else 0

    # Compute residual sd for simulation
    preds = best_model.predict(dummies)
    resid_var = np.average((df_stat[stat] - preds) ** 2, weights=weights)
    sd = np.sqrt(resid_var)
    resid = df_stat[stat] - preds

    # Approximate weather coefs with importances
    temp_coef = importances.loc[importances['coef_name'] == 'temp', 'importance'].values[0] if 'temp' in importances['coef_name'].values else 0
    wind_coef = importances.loc[importances['coef_name'] == 'wind', 'importance'].values[0] if 'wind' in importances['coef_name'].values else 0
    roof_coefs = {row['coef_name']: row['importance'] for _, row in importances[importances['coef_name'].str.startswith('roof_')].iterrows()}

    return off_df.set_index('team'), def_df.set_index('team'), hfa_val, sd, 0, best_model, resid, weights, temp_coef, wind_coef, roof_coefs, dummies.columns.tolist()

def get_points_adjusted_ratings(schedules):
    # Create game level for points
    home_df = schedules[['game_id', 'home_team', 'away_team', 'home_score', 'away_score', 'location', 'season', 'gameday', 'roof', 'temp', 'wind']].copy()
    home_df['neutral_site'] = (home_df['location'] == 'Neutral')
    home_df['offense'] = home_df['home_team']
    home_df['defense'] = home_df['away_team']
    home_df['stat'] = home_df['home_score']
    home_df['hfa'] = 1
    home_df.loc[home_df['neutral_site'], 'hfa'] = 0
    home_df['game_date'] = pd.to_datetime(home_df['gameday'])
    away_df = home_df.copy()
    away_df['offense'] = home_df['away_team']
    away_df['defense'] = home_df['home_team']
    away_df['stat'] = home_df['away_score']
    away_df['hfa'] = -home_df['hfa']
    df_points = pd.concat([home_df, away_df])[['offense', 'hfa', 'defense', 'stat', 'game_date', 'roof', 'temp', 'wind']].dropna()
    return get_adjusted_ratings(df_points, 'stat')

def simulate_matchup(team1, team2, team1_home=True, n_sims=10000, roof='outdoors', temp=70, wind=5):
    # Assume years for data
    years = list(range(2020, 2026))  # Adjust as needed
    pbp = nfl.load_pbp(seasons=years)
    pbp = pbp.to_pandas()
    schedules = nfl.load_schedules(seasons=years)
    schedules = schedules.to_pandas()
    game_off = compute_game_level_stats(pbp, schedules)

    # Stats to project
    stats = ['total_yards', 'passing_yards', 'rushing_yards', 'total_plays', 'pass_attempt', 'rush_attempt', 'total_epa', 'sacks_allowed', 'turnovers', 'success_rate']

    # Drop NA for multivariate consistency
    game_off = game_off.dropna(subset=stats + ['game_date'])

    adjusted_off = {}
    adjusted_def = {}
    hfa_dict = {}
    sd_dict = {}
    intercept_dict = {}
    model_dict = {}
    resid_dict = {}
    weights = None  # Will be set from first stat
    temp_coef_dict = {}
    wind_coef_dict = {}
    roof_coefs_dict = {}
    dummy_cols_dict = {}

    for s in stats:
        off, deff, hfa, sd, intercept, model, resid, w, temp_coef, wind_coef, roof_coefs, dummy_cols = get_adjusted_ratings(game_off, s)
        adjusted_off[s] = off
        adjusted_def[s] = deff
        hfa_dict[s] = hfa
        sd_dict[s] = sd
        intercept_dict[s] = intercept
        model_dict[s] = model
        resid_dict[s] = resid
        temp_coef_dict[s] = temp_coef
        wind_coef_dict[s] = wind_coef
        roof_coefs_dict[s] = roof_coefs
        dummy_cols_dict[s] = dummy_cols
        if weights is None:
            weights = w  # Assume same for all

    # Compute covariance from residuals
    resid_df = pd.DataFrame(resid_dict)
    cov_matrix = weighted_cov(resid_df.values, weights)

    # Points separate
    off_points, def_points, hfa_points, sd_points, intercept_points, model_points, resid_points, weights_points, temp_coef_points, wind_coef_points, roof_coefs_points, dummy_cols_points = get_points_adjusted_ratings(schedules)

    # Projections
    projections_team1 = {}
    projections_team2 = {}
    hfa_sign_team1_off = 1 if team1_home else -1
    hfa_sign_team2_off = -hfa_sign_team1_off

    sims_points_team1 = None
    sims_points_team2 = None

    # Handle 'indoors' as 'dome'
    if roof == 'indoors':
        roof = 'dome'

    # Multivariate sim for stats
    proj_team1 = {}
    proj_team2 = {}
    for s in stats:
        model = model_dict[s]
        dummy_cols = dummy_cols_dict[s]
        # Create dummy row for team1 offense vs team2 defense
        matchup_df_team1 = pd.DataFrame(columns=dummy_cols)
        if 'offense_' + team1 in matchup_df_team1.columns:
            matchup_df_team1.loc[0, 'offense_' + team1] = 1
        if 'defense_' + team2 in matchup_df_team1.columns:
            matchup_df_team1.loc[0, 'defense_' + team2] = 1
        matchup_df_team1.loc[0, 'hfa'] = hfa_sign_team1_off
        roof_column = 'roof_' + roof
        if roof_column in matchup_df_team1.columns:
            matchup_df_team1.loc[0, roof_column] = 1
        matchup_df_team1.loc[0, 'temp'] = temp
        matchup_df_team1.loc[0, 'wind'] = wind
        matchup_df_team1 = matchup_df_team1.fillna(0)
        proj_team1[s] = model.predict(matchup_df_team1)[0]

        # For team2
        matchup_df_team2 = pd.DataFrame(columns=dummy_cols)
        if 'offense_' + team2 in matchup_df_team2.columns:
            matchup_df_team2.loc[0, 'offense_' + team2] = 1
        if 'defense_' + team1 in matchup_df_team2.columns:
            matchup_df_team2.loc[0, 'defense_' + team1] = 1
        matchup_df_team2.loc[0, 'hfa'] = hfa_sign_team2_off
        if roof_column in matchup_df_team2.columns:
            matchup_df_team2.loc[0, roof_column] = 1
        matchup_df_team2.loc[0, 'temp'] = temp
        matchup_df_team2.loc[0, 'wind'] = wind
        matchup_df_team2 = matchup_df_team2.fillna(0)
        proj_team2[s] = model.predict(matchup_df_team2)[0]

    mean_team1 = [proj_team1[s] for s in stats]
    mean_team2 = [proj_team2[s] for s in stats]

    sims_team1_multi = multivariate_normal.rvs(mean=mean_team1, cov=cov_matrix, size=n_sims)
    sims_team2_multi = multivariate_normal.rvs(mean=mean_team2, cov=cov_matrix, size=n_sims)

    for i, s in enumerate(stats):
        projections_team1[s] = np.mean(sims_team1_multi[:, i])
        projections_team1[s + '_sd'] = np.std(sims_team1_multi[:, i])
        projections_team2[s] = np.mean(sims_team2_multi[:, i])
        projections_team2[s + '_sd'] = np.std(sims_team2_multi[:, i])

    # Points simulation (univariate)
    s = 'points'
    dummy_cols = dummy_cols_points
    model = model_points
    # Team1
    matchup_df_team1 = pd.DataFrame(columns=dummy_cols)
    if 'offense_' + team1 in matchup_df_team1.columns:
        matchup_df_team1.loc[0, 'offense_' + team1] = 1
    if 'defense_' + team2 in matchup_df_team1.columns:
        matchup_df_team1.loc[0, 'defense_' + team2] = 1
    matchup_df_team1.loc[0, 'hfa'] = hfa_sign_team1_off
    roof_column = 'roof_' + roof
    if roof_column in matchup_df_team1.columns:
        matchup_df_team1.loc[0, roof_column] = 1
    matchup_df_team1.loc[0, 'temp'] = temp
    matchup_df_team1.loc[0, 'wind'] = wind
    matchup_df_team1 = matchup_df_team1.fillna(0)
    proj_team1_points = model.predict(matchup_df_team1)[0]

    # Team2
    matchup_df_team2 = pd.DataFrame(columns=dummy_cols)
    if 'offense_' + team2 in matchup_df_team2.columns:
        matchup_df_team2.loc[0, 'offense_' + team2] = 1
    if 'defense_' + team1 in matchup_df_team2.columns:
        matchup_df_team2.loc[0, 'defense_' + team1] = 1
    matchup_df_team2.loc[0, 'hfa'] = hfa_sign_team2_off
    if roof_column in matchup_df_team2.columns:
        matchup_df_team2.loc[0, roof_column] = 1
    matchup_df_team2.loc[0, 'temp'] = temp
    matchup_df_team2.loc[0, 'wind'] = wind
    matchup_df_team2 = matchup_df_team2.fillna(0)
    proj_team2_points = model.predict(matchup_df_team2)[0]

    sims_points_team1 = norm.rvs(loc=proj_team1_points, scale=sd_points, size=n_sims)
    sims_points_team2 = norm.rvs(loc=proj_team2_points, scale=sd_points, size=n_sims)

    projections_team1['points'] = np.mean(sims_points_team1)
    projections_team1['points_sd'] = np.std(sims_points_team1)
    projections_team2['points'] = np.mean(sims_points_team2)
    projections_team2['points_sd'] = np.std(sims_points_team2)

    # Derive defensive metrics from opponent
    projections_team1['sacks'] = projections_team2['sacks_allowed']
    projections_team1['sacks_sd'] = projections_team2['sacks_allowed_sd']
    projections_team2['sacks'] = projections_team1['sacks_allowed']
    projections_team2['sacks_sd'] = projections_team1['sacks_allowed_sd']

    projections_team1['turnovers_forced'] = projections_team2['turnovers']
    projections_team1['turnovers_forced_sd'] = projections_team2['turnovers_sd']
    projections_team2['turnovers_forced'] = projections_team1['turnovers']
    projections_team2['turnovers_forced_sd'] = projections_team1['turnovers_sd']

    # EPA allowed is opponent EPA
    projections_team1['epa_allowed'] = projections_team2['total_epa']
    projections_team1['epa_allowed_sd'] = projections_team2['total_epa_sd']
    projections_team2['epa_allowed'] = projections_team1['total_epa']
    projections_team2['epa_allowed_sd'] = projections_team1['total_epa_sd']

    # Points allowed
    projections_team1['points_allowed'] = projections_team2['points']
    projections_team1['points_allowed_sd'] = projections_team2['points_sd']
    projections_team2['points_allowed'] = projections_team1['points']
    projections_team2['points_allowed_sd'] = projections_team1['points_sd']

    # Total yards allowed
    projections_team1['total_yards_allowed'] = projections_team2['total_yards']
    projections_team1['total_yards_allowed_sd'] = projections_team2['total_yards_sd']
    projections_team2['total_yards_allowed'] = projections_team1['total_yards']
    projections_team2['total_yards_allowed_sd'] = projections_team1['total_yards_sd']

    # Etc. for passing, rushing allowed
    projections_team1['passing_yards_allowed'] = projections_team2['passing_yards']
    projections_team1['passing_yards_allowed_sd'] = projections_team2['passing_yards_sd']
    projections_team2['passing_yards_allowed'] = projections_team1['passing_yards']
    projections_team2['passing_yards_allowed_sd'] = projections_team1['passing_yards_sd']
    projections_team1['rushing_yards_allowed'] = projections_team2['rushing_yards']
    projections_team1['rushing_yards_allowed_sd'] = projections_team2['rushing_yards_sd']
    projections_team2['rushing_yards_allowed'] = projections_team1['rushing_yards']
    projections_team2['rushing_yards_allowed_sd'] = projections_team1['rushing_yards_sd']

    # Win probability
    if sims_points_team1 is not None and sims_points_team2 is not None:
        win_team1 = np.mean(sims_points_team1 > sims_points_team2)
        tie = np.mean(sims_points_team1 == sims_points_team2)
        win_prob_team1 = win_team1 + 0.5 * tie
        projections_team1['win_probability'] = win_prob_team1
        projections_team2['win_probability'] = 1 - win_prob_team1

    # Output
    print(f"Projections for {team1} (home: {team1_home}) vs {team2}:")
    print(pd.Series(projections_team1).to_frame(team1))
    print(pd.Series(projections_team2).to_frame(team2))

from sklearn.metrics import mean_absolute_error, mean_squared_error

def backtest_model(train_years, test_year):
    # Load train data
    pbp_train = nfl.load_pbp(seasons=train_years).to_pandas()
    schedules_train = nfl.load_schedules(seasons=train_years).to_pandas()
    
    # Load test schedules (for actual outcomes and game details)
    schedules_test = nfl.load_schedules(seasons=[test_year]).to_pandas()
    completed_games = schedules_test[schedules_test['result'].notna()].copy()  # Only completed games
    
    # To use train data only, override global data loads in functions (hack: pass as params or monkey-patch)
    # For simplicity, assume you modify simulate_matchup to accept pbp/schedules params
    # Here, assuming you add params to simulate_matchup(pbp=pbp_train, schedules=schedules_train, ...)
    
    proj_points = []  # List of (team1_proj_points, team2_proj_points)
    actual_points = []  # List of (home_score, away_score)
    proj_win_probs = []  # List of team1 win prob
    actual_wins = []  # 1 if team1 win, 0.5 tie, 0 loss
    
    for _, game in completed_games.iterrows():
        team1 = game['home_team']
        team2 = game['away_team']
        team1_home = True
        # Get historical weather/roof for accuracy (or default)
        roof = game['roof'] if pd.notna(game['roof']) else 'outdoors'
        temp = game['temp'] if pd.notna(game['temp']) else 70
        wind = game['wind'] if pd.notna(game['wind']) else 5
        
        # Run simulation with train data (modify simulate_matchup to use pbp_train, schedules_train)
        # projections_team1, projections_team2 = simulate_matchup(team1, team2, team1_home, n_sims=1000, roof=roof, temp=temp, wind=wind, pbp=pbp_train, schedules=schedules_train)
        # For now, mock - replace with actual call
        projections_team1 = {'points': np.random.uniform(20, 30), 'win_probability': np.random.uniform(0.4, 0.6)}  # Mock
        projections_team2 = {'points': np.random.uniform(20, 30)}
        
        proj_points.append((projections_team1['points'], projections_team2['points']))
        actual_points.append((game['home_score'], game['away_score']))
        
        proj_win_probs.append(projections_team1['win_probability'])
        if game['result'] > 0:
            actual_wins.append(1)  # Home win
        elif game['result'] < 0:
            actual_wins.append(0)  # Away win
        else:
            actual_wins.append(0.5)  # Tie
    
    proj_points = np.array(proj_points)
    actual_points = np.array(actual_points)
    proj_win_probs = np.array(proj_win_probs)
    actual_wins = np.array(actual_wins)
    
    # Metrics
    mae_home = mean_absolute_error(actual_points[:,0], proj_points[:,0])
    mae_away = mean_absolute_error(actual_points[:,1], proj_points[:,1])
    rmse_home = np.sqrt(mean_squared_error(actual_points[:,0], proj_points[:,0]))
    rmse_away = np.sqrt(mean_squared_error(actual_points[:,1], proj_points[:,1]))
    brier_score = np.mean((proj_win_probs - actual_wins) ** 2)
    
    results = {
        'Num Games': len(completed_games),
        'MAE Home Points': mae_home,
        'MAE Away Points': mae_away,
        'RMSE Home Points': rmse_home,
        'RMSE Away Points': rmse_away,
        'Brier Score (Win Prob)': brier_score
    }
    
    return results

# Example call (add to bottom of script)
if __name__ == "__main__":
    backtest_results = backtest_model(train_years=[2020, 2021, 2022, 2023], test_year=2024)
    print(backtest_results)

# Example usage
# simulate_matchup('KC', 'SF', team1_home=True, roof='outdoors', temp=60, wind=10)
#  simulate_matchup('DAL', 'KC', team1_home=True, roof='indoors', temp=60, wind=0)