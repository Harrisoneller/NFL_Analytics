import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import nflreadpy as nfl  # Note: Consider migrating to nfl_data_py for future compatibility
from scipy.stats import norm, multivariate_normal, poisson, nbinom  # For simulation

def weighted_cov(X, w):
    X = np.asarray(X)
    w = np.asarray(w)
    m = np.average(X, axis=0, weights=w)
    X_centered = X - m
    cov = np.dot((X_centered.T * w), X_centered) / w.sum()
    return cov

def compute_game_level_stats(pbp, schedules):
    # Filter to offensive plays
    pbp_off = pbp[(pbp['pass_attempt'] == 1) | (pbp['rush_attempt'] == 1)].copy()  # Add copy to avoid SettingWithCopyWarning
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

    # Enhanced features
    # Red zone specific (red_zone as yardline_100 <=20)
    pbp_off['in_red_zone'] = (pbp_off['yardline_100'] <= 20).astype(int)
    pbp_off['red_zone_td'] = ((pbp_off['pass_touchdown'] == 1) | (pbp_off['rush_touchdown'] == 1)) & (pbp_off['in_red_zone'] == 1)
    red_zone = pbp_off.groupby(['game_id', 'posteam']).agg({
        'in_red_zone': 'sum',  # Attempts
        'red_zone_td': 'sum'   # TDs
    }).reset_index().rename(columns={
        'posteam': 'offense',
        'in_red_zone': 'red_zone_attempts',
        'red_zone_td': 'red_zone_tds'
    })

    # 3rd down
    third_down_plays = pbp_off[pbp_off['down'] == 3]
    third_down = third_down_plays.groupby(['game_id', 'posteam']).agg({
        'third_down_converted': 'sum',
        'third_down_failed': 'sum'
    }).reset_index().rename(columns={'posteam': 'offense'})

    # Explosive plays (20+ yards)
    explosive = pbp_off.groupby(['game_id', 'posteam']).agg({
        'yards_gained': lambda x: (x >= 20).sum()  # Count of 20+ yd plays
    }).reset_index().rename(columns={
        'posteam': 'offense',
        'yards_gained': 'explosive_plays'
    })

    # Pressure allowed (offense view)
    pressure = pbp_off[pbp_off['pass_attempt'] == 1].groupby(['game_id', 'posteam']).agg({
        'qb_hit': 'sum'
    }).reset_index().rename(columns={'posteam': 'offense'})

    # Merge enhanced
    game_off = game_off.merge(red_zone, on=['game_id', 'offense'], how='left').fillna(0)
    game_off = game_off.merge(third_down, on=['game_id', 'offense'], how='left').fillna(0)
    game_off = game_off.merge(explosive, on=['game_id', 'offense'], how='left').fillna(0)
    game_off = game_off.merge(pressure, on=['game_id', 'offense'], how='left').fillna(0)

    # Compute rates
    game_off['red_zone_td_rate'] = np.where(game_off['red_zone_attempts'] > 0, game_off['red_zone_tds'] / game_off['red_zone_attempts'], 0)
    total_third_downs = game_off['third_down_converted'] + game_off['third_down_failed']
    game_off['third_down_conv_rate'] = np.where(total_third_downs > 0, game_off['third_down_converted'] / total_third_downs, 0)
    game_off['explosive_rate'] = np.where(game_off['total_plays'] > 0, game_off['explosive_plays'] / game_off['total_plays'], 0)
    game_off['pressure_allowed_rate'] = np.where(game_off['pass_attempt'] > 0, (game_off['qb_hit'] + game_off['sacks_allowed']) / game_off['pass_attempt'], 0)

    # Feature interactions (example: pass_epa * pressure_allowed_rate, success_rate * third_down_conv_rate)
    game_off['pass_epa_pressure_int'] = game_off['pass_epa'] * game_off['pressure_allowed_rate']
    game_off['success_third_int'] = game_off['success_rate'] * game_off['third_down_conv_rate']

    # New: Rest days (days since last game for offense)
    schedules['gameday'] = pd.to_datetime(schedules['gameday'])
    # For each team, get all games (home or away)
    team_games = pd.melt(schedules, id_vars=['game_id', 'gameday'], value_vars=['home_team', 'away_team'], value_name='team').sort_values(['team', 'gameday'])
    team_games['rest_days'] = team_games.groupby('team')['gameday'].diff().dt.days.fillna(7)  # Default 7 for first game
    # Merge back to game_off (for offense)
    game_off = game_off.merge(team_games[['game_id', 'team', 'rest_days']], left_on=['game_id', 'offense'], right_on=['game_id', 'team'], how='left').drop('team', axis=1)

    return game_off

def get_adjusted_ratings(df, stat, alpha_values=[0, 0.1, 1, 5]):
    df_stat = df[['offense', 'hfa', 'defense', stat, 'game_date', 'roof', 'temp', 'wind', 'rest_days']].dropna().copy()  # Add rest_days
    df_stat['game_date'] = pd.to_datetime(df_stat['game_date'])
    max_date = df_stat['game_date'].max()
    days_ago = (max_date - df_stat['game_date']).dt.days
    decay_rate = np.log(2) / 180  # Half-life of 180 days
    weights = np.exp(-decay_rate * days_ago)
    # Handle weather
    df_stat['temp'] = df_stat['temp'].fillna(72)  # Assume indoor temp
    df_stat['wind'] = df_stat['wind'].fillna(0)  # No wind indoors
    dummies = pd.get_dummies(df_stat[['offense', 'hfa', 'defense', 'roof']])
    dummies = pd.concat([dummies, df_stat[['temp', 'wind', 'rest_days']]], axis=1)  # Add rest_days as numeric
    # XGBoost part
    param_grid = {
        'reg_alpha': alpha_values,
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0]
    }
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(dummies, df_stat[stat], sample_weight=weights)
    best_xgb = grid_search.best_estimator_
    # Ridge part for ensemble
    ridge = RidgeCV(alphas=[0.1, 1, 10], cv=3).fit(dummies, df_stat[stat], sample_weight=weights)
    # Ensemble predictions (tuned weights: more on XGBoost)
    preds_xgb = best_xgb.predict(dummies)
    preds_ridge = ridge.predict(dummies)
    preds = 0.8 * preds_xgb + 0.2 * preds_ridge  # Tuned to 0.8/0.2
    # Approximate ratings using XGBoost feature importances (primary model)
    importances = pd.DataFrame({'coef_name': dummies.columns, 'importance': best_xgb.feature_importances_})
    off_df = importances[importances['coef_name'].str.startswith('offense_')].copy()
    off_df['team'] = off_df['coef_name'].str.replace('offense_', '')
    off_df = off_df[['team', 'importance']].rename(columns={'importance': f'adj_off_{stat}'})
    def_df = importances[importances['coef_name'].str.startswith('defense_')].copy()
    def_df['team'] = def_df['coef_name'].str.replace('defense_', '')
    def_df = def_df[['team', 'importance']].rename(columns={'importance': f'adj_def_{stat}'})
    hfa_val = importances.loc[importances['coef_name'] == 'hfa', 'importance'].values[0] if 'hfa' in importances['coef_name'].values else 0
    # Compute residual sd for simulation (using ensemble preds)
    resid_var = np.average((df_stat[stat] - preds) ** 2, weights=weights)
    sd = np.sqrt(resid_var)
    resid = df_stat[stat] - preds
    # Approximate weather coefs with importances
    temp_coef = importances.loc[importances['coef_name'] == 'temp', 'importance'].values[0] if 'temp' in importances['coef_name'].values else 0
    wind_coef = importances.loc[importances['coef_name'] == 'wind', 'importance'].values[0] if 'wind' in importances['coef_name'].values else 0
    roof_coefs = {row['coef_name']: row['importance'] for _, row in importances[importances['coef_name'].str.startswith('roof_')].iterrows()}
    return off_df.set_index('team'), def_df.set_index('team'), hfa_val, sd, 0, (best_xgb, ridge), resid, weights, temp_coef, wind_coef, roof_coefs, dummies.columns.tolist()

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
    df_points = pd.concat([home_df, away_df])[['game_id', 'offense', 'hfa', 'defense', 'stat', 'game_date', 'roof', 'temp', 'wind']].dropna()  # Added game_id for pairing
    # New: Add rest_days for points df
    team_games = pd.melt(schedules, id_vars=['game_id', 'gameday'], value_vars=['home_team', 'away_team'], value_name='team').sort_values(['team', 'gameday'])
    team_games['rest_days'] = team_games.groupby('team')['gameday'].diff().dt.days.fillna(7)
    df_points = df_points.merge(team_games[['game_id', 'team', 'rest_days']], left_on=['game_id', 'offense'], right_on=['game_id', 'team'], how='left').drop('team', axis=1)
    # Get ratings as before (now returns tuple for model)
    off, deff, hfa, sd, intercept, model_tuple, resid, weights, temp_coef, wind_coef, roof_coefs, dummy_cols = get_adjusted_ratings(df_points, 'stat')
    # New: Compute paired residuals for covariance
    dummies_points = pd.concat([pd.get_dummies(df_points[['offense', 'hfa', 'defense', 'roof']]), df_points[['temp', 'wind', 'rest_days']]], axis=1).reindex(columns=dummy_cols, fill_value=0)
    best_xgb, ridge = model_tuple
    preds_xgb = best_xgb.predict(dummies_points)
    preds_ridge = ridge.predict(dummies_points)
    preds = 0.8 * preds_xgb + 0.2 * preds_ridge
    df_points['pred'] = preds
    df_points['resid'] = df_points['stat'] - df_points['pred']
    # Pair home/away residuals
    home_resid = df_points[df_points['hfa'] > 0].set_index('game_id')['resid']
    away_resid = df_points[df_points['hfa'] < 0].set_index('game_id')['resid']
    paired_resid = pd.concat([home_resid, away_resid], axis=1, keys=['home_resid', 'away_resid']).dropna()
    # Use weights from home (assuming symmetric)
    weights_paired = weights[df_points['hfa'] > 0][:len(paired_resid)]
    cov_matrix_points = weighted_cov(paired_resid.values, weights_paired)
    return off, deff, hfa, sd, intercept, model_tuple, resid, weights, temp_coef, wind_coef, roof_coefs, dummy_cols, cov_matrix_points

def project_matchup(team1, team2, team1_home, roof, temp, wind, stats, model_dict, dummy_cols_dict, cov_matrix, model_points, dummy_cols_points, cov_matrix_points, n_sims=10000, iso_calibrator=None):
    projections_team1 = {}
    projections_team2 = {}
    hfa_sign_team1_off = 1 if team1_home else -1
    hfa_sign_team2_off = -hfa_sign_team1_off
    # Handle 'indoors' as 'dome'
    if roof == 'indoors':
        roof = 'dome'
    # Multivariate sim for stats
    proj_team1 = {}
    proj_team2 = {}
    for s in stats:
        model_tuple = model_dict[s]
        dummy_cols = dummy_cols_dict[s]
        # Create dummy row for team1 offense vs team2 defense
        matchup_df_team1 = pd.DataFrame(columns=dummy_cols, index=[0]).fillna(0)
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
        best_xgb, ridge = model_tuple
        pred_xgb = best_xgb.predict(matchup_df_team1)[0]
        pred_ridge = ridge.predict(matchup_df_team1)[0]
        proj_team1[s] = 0.8 * pred_xgb + 0.2 * pred_ridge
        # For team2
        matchup_df_team2 = pd.DataFrame(columns=dummy_cols, index=[0]).fillna(0)
        if 'offense_' + team2 in matchup_df_team2.columns:
            matchup_df_team2.loc[0, 'offense_' + team2] = 1
        if 'defense_' + team1 in matchup_df_team2.columns:
            matchup_df_team2.loc[0, 'defense_' + team1] = 1
        matchup_df_team2.loc[0, 'hfa'] = hfa_sign_team2_off
        if roof_column in matchup_df_team2.columns:
            matchup_df_team2.loc[0, roof_column] = 1
        matchup_df_team2.loc[0, 'temp'] = temp
        matchup_df_team2.loc[0, 'wind'] = wind
        pred_xgb = best_xgb.predict(matchup_df_team2)[0]
        pred_ridge = ridge.predict(matchup_df_team2)[0]
        proj_team2[s] = 0.8 * pred_xgb + 0.2 * pred_ridge
    mean_team1 = [proj_team1[s] for s in stats]
    mean_team2 = [proj_team2[s] for s in stats]
    sims_team1_multi = multivariate_normal.rvs(mean=mean_team1, cov=cov_matrix, size=n_sims)
    sims_team2_multi = multivariate_normal.rvs(mean=mean_team2, cov=cov_matrix, size=n_sims)
    for i, s in enumerate(stats):
        projections_team1[s] = np.mean(sims_team1_multi[:, i])
        projections_team1[s + '_sd'] = np.std(sims_team1_multi[:, i])
        projections_team2[s] = np.mean(sims_team2_multi[:, i])
        projections_team2[s + '_sd'] = np.std(sims_team2_multi[:, i])
    # Points simulation (negative binomial for overdispersion)
    dummy_cols = dummy_cols_points
    best_xgb, ridge = model_points
    # Team1
    matchup_df_team1 = pd.DataFrame(columns=dummy_cols, index=[0]).fillna(0)
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
    pred_xgb = best_xgb.predict(matchup_df_team1)[0]
    pred_ridge = ridge.predict(matchup_df_team1)[0]
    proj_team1_points = 0.8 * pred_xgb + 0.2 * pred_ridge
    # Team2
    matchup_df_team2 = pd.DataFrame(columns=dummy_cols, index=[0]).fillna(0)
    if 'offense_' + team2 in matchup_df_team2.columns:
        matchup_df_team2.loc[0, 'offense_' + team2] = 1
    if 'defense_' + team1 in matchup_df_team2.columns:
        matchup_df_team2.loc[0, 'defense_' + team1] = 1
    matchup_df_team2.loc[0, 'hfa'] = hfa_sign_team2_off
    if roof_column in matchup_df_team2.columns:
        matchup_df_team2.loc[0, roof_column] = 1
    matchup_df_team2.loc[0, 'temp'] = temp
    matchup_df_team2.loc[0, 'wind'] = wind
    pred_xgb = best_xgb.predict(matchup_df_team2)[0]
    pred_ridge = ridge.predict(matchup_df_team2)[0]
    proj_team2_points = 0.8 * pred_xgb + 0.2 * pred_ridge
    # Negative binomial simulation (overdispersion with scale=2)
    scale = 2  # Tune based on historical variance
    sims_points_team1 = nbinom.rvs(n=proj_team1_points / scale, p=1 / (1 + scale), size=n_sims).astype(float)  # Cast to float
    sims_points_team2 = nbinom.rvs(n=proj_team2_points / scale, p=1 / (1 + scale), size=n_sims).astype(float)
    # Add correlation (shared noise, corr~0.2)
    shared_noise = np.random.normal(0, 1, n_sims) * 0.2 * np.sqrt(proj_team1_points * proj_team2_points)  # Approximate cov
    sims_points_team1 += shared_noise
    sims_points_team2 += shared_noise
    sims_points_team1 = np.maximum(0, np.round(sims_points_team1)).astype(int)  # Round and clip
    sims_points_team2 = np.maximum(0, np.round(sims_points_team2)).astype(int)
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
    # Passing, rushing allowed
    projections_team1['passing_yards_allowed'] = projections_team2['passing_yards']
    projections_team1['passing_yards_allowed_sd'] = projections_team2['passing_yards_sd']
    projections_team2['passing_yards_allowed'] = projections_team1['passing_yards']
    projections_team2['passing_yards_allowed_sd'] = projections_team1['passing_yards_sd']
    projections_team1['rushing_yards_allowed'] = projections_team2['rushing_yards']
    projections_team1['rushing_yards_allowed_sd'] = projections_team2['rushing_yards_sd']
    projections_team2['rushing_yards_allowed'] = projections_team1['rushing_yards']
    projections_team2['rushing_yards_allowed_sd'] = projections_team1['rushing_yards_sd']
    # Enhanced defensive derivations
    projections_team1['red_zone_td_allowed_rate'] = projections_team2['red_zone_td_rate']
    projections_team1['red_zone_td_allowed_rate_sd'] = projections_team2['red_zone_td_rate_sd']
    projections_team2['red_zone_td_allowed_rate'] = projections_team1['red_zone_td_rate']
    projections_team2['red_zone_td_allowed_rate_sd'] = projections_team1['red_zone_td_rate_sd']
    projections_team1['third_down_conv_allowed_rate'] = projections_team2['third_down_conv_rate']
    projections_team1['third_down_conv_allowed_rate_sd'] = projections_team2['third_down_conv_rate_sd']
    projections_team2['third_down_conv_allowed_rate'] = projections_team1['third_down_conv_rate']
    projections_team2['third_down_conv_allowed_rate_sd'] = projections_team1['third_down_conv_rate_sd']
    projections_team1['explosive_allowed_rate'] = projections_team2['explosive_rate']
    projections_team1['explosive_allowed_rate_sd'] = projections_team2['explosive_rate_sd']
    projections_team2['explosive_allowed_rate'] = projections_team1['explosive_rate']
    projections_team2['explosive_allowed_rate_sd'] = projections_team1['explosive_rate_sd']
    projections_team1['pressure_rate'] = projections_team2['pressure_allowed_rate']
    projections_team1['pressure_rate_sd'] = projections_team2['pressure_allowed_rate_sd']
    projections_team2['pressure_rate'] = projections_team1['pressure_allowed_rate']
    projections_team2['pressure_rate_sd'] = projections_team1['pressure_allowed_rate_sd']
    # Win probability
    win_team1_raw = np.mean(sims_points_team1 > sims_points_team2) + 0.5 * np.mean(sims_points_team1 == sims_points_team2)
    # Calibrate if isolator provided
    if iso_calibrator is not None:
        win_prob_team1 = iso_calibrator.predict([win_team1_raw])[0]
    else:
        win_prob_team1 = win_team1_raw
    projections_team1['win_probability'] = win_prob_team1
    projections_team2['win_probability'] = 1 - win_prob_team1
    return projections_team1, projections_team2

def simulate_matchup(team1, team2, team1_home=True, n_sims=10000, roof='outdoors', temp=70, wind=5, pbp=None, schedules=None, iso_calibrator=None):
    if pbp is None or schedules is None:
        years = list(range(2015, 2026))  # Updated: Extended back to 2015 for more data
        pbp = nfl.load_pbp(seasons=years).to_pandas()
        schedules = nfl.load_schedules(seasons=years).to_pandas()
    game_off = compute_game_level_stats(pbp, schedules)
    # Stats to project (enhanced)
    stats = ['total_yards', 'passing_yards', 'rushing_yards', 'total_plays', 'pass_attempt', 'rush_attempt', 'total_epa', 'sacks_allowed', 'turnovers', 'success_rate',
             'red_zone_td_rate', 'third_down_conv_rate', 'explosive_rate', 'pressure_allowed_rate', 'pass_epa_pressure_int', 'success_third_int']
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
        off, deff, hfa, sd, intercept, model_tuple, resid, w, temp_coef, wind_coef, roof_coefs, dummy_cols = get_adjusted_ratings(game_off, s)
        adjusted_off[s] = off
        adjusted_def[s] = deff
        hfa_dict[s] = hfa
        sd_dict[s] = sd
        intercept_dict[s] = intercept
        model_dict[s] = model_tuple
        resid_dict[s] = resid
        temp_coef_dict[s] = temp_coef
        wind_coef_dict[s] = wind_coef
        roof_coefs_dict[s] = roof_coefs
        dummy_cols_dict[s] = dummy_cols
        if weights is None:
            weights = w  # Assume same for all
    # Compute covariance from residuals (stratify by roof for example)
    resid_df = pd.DataFrame(resid_dict)
    cov_matrix = {}
    for roof_type in game_off['roof'].unique():
        mask = game_off['roof'] == roof_type
        cov_matrix[roof_type] = weighted_cov(resid_df[mask].values, weights[mask]) if mask.sum() > 0 else weighted_cov(resid_df.values, weights)
    # Points separate (updated to return cov_matrix_points)
    off_points, def_points, hfa_points, sd_points, intercept_points, model_points, resid_points, weights_points, temp_coef_points, wind_coef_points, roof_coefs_points, dummy_cols_points, cov_matrix_points = get_points_adjusted_ratings(schedules)
    # Now project (pass roof-specific cov if available)
    used_cov = cov_matrix.get(roof, cov_matrix['outdoors'])  # Default to outdoors
    projections_team1, projections_team2 = project_matchup(team1, team2, team1_home, roof, temp, wind, stats, model_dict, dummy_cols_dict, used_cov, model_points, dummy_cols_points, cov_matrix_points, n_sims, iso_calibrator)
    # Output
    print(f"Projections for {team1} (home: {team1_home}) vs {team2}:")
    print(pd.Series(projections_team1).to_frame(team1))
    print(pd.Series(projections_team2).to_frame(team2))
    return projections_team1, projections_team2

from sklearn.metrics import mean_absolute_error, mean_squared_error

def backtest_model(train_years, test_year):
    # Load train data
    pbp_train = nfl.load_pbp(seasons=train_years).to_pandas()
    schedules_train = nfl.load_schedules(seasons=train_years).to_pandas()
    # Precompute models and cov matrices
    game_off = compute_game_level_stats(pbp_train, schedules_train)
    stats = ['total_yards', 'passing_yards', 'rushing_yards', 'total_plays', 'pass_attempt', 'rush_attempt', 'total_epa', 'sacks_allowed', 'turnovers', 'success_rate',
             'red_zone_td_rate', 'third_down_conv_rate', 'explosive_rate', 'pressure_allowed_rate', 'pass_epa_pressure_int', 'success_third_int']
    game_off = game_off.dropna(subset=stats + ['game_date'])
    model_dict = {}
    dummy_cols_dict = {}
    resid_dict = {}
    weights = None
    for s in stats:
        _, _, _, _, _, model_tuple, resid, w, _, _, _, dummy_cols = get_adjusted_ratings(game_off, s)
        model_dict[s] = model_tuple
        dummy_cols_dict[s] = dummy_cols
        resid_dict[s] = resid
        if weights is None:
            weights = w
    resid_df = pd.DataFrame(resid_dict)
    cov_matrix = {}
    for roof_type in game_off['roof'].unique():
        mask = game_off['roof'] == roof_type
        cov_matrix[roof_type] = weighted_cov(resid_df[mask].values, weights[mask]) if mask.sum() > 0 else weighted_cov(resid_df.values, weights)
    # Points
    _, _, _, _, _, model_points, _, _, _, _, _, dummy_cols_points, cov_matrix_points = get_points_adjusted_ratings(schedules_train)
    # Load test schedules (for actual outcomes and game details)
    schedules_test = nfl.load_schedules(seasons=[test_year]).to_pandas()
    completed_games = schedules_test[schedules_test['result'].notna()].copy()  # Only completed games
    proj_points = []  # List of (team1_proj_points, team2_proj_points)
    actual_points = []  # List of (home_score, away_score)
    proj_win_probs_raw = []  # For calibration
    actual_wins = []  # 1 if team1 win, 0.5 tie, 0 loss
    for _, game in completed_games.iterrows():
        team1 = game['home_team']
        team2 = game['away_team']
        team1_home = True
        # Get historical weather/roof for accuracy (or default)
        roof = game['roof'] if pd.notna(game['roof']) else 'outdoors'
        temp = game['temp'] if pd.notna(game['temp']) else 70
        wind = game['wind'] if pd.notna(game['wind']) else 5
        # Run projection (no training)
        used_cov = cov_matrix.get(roof, cov_matrix['outdoors'])
        projections_team1, projections_team2 = project_matchup(
            team1, team2, team1_home, roof, temp, wind, stats, model_dict, dummy_cols_dict, used_cov, model_points, dummy_cols_points, cov_matrix_points, n_sims=1000
        )
        proj_points.append((projections_team1['points'], projections_team2['points']))
        actual_points.append((game['home_score'], game['away_score']))
        proj_win_probs_raw.append(projections_team1['win_probability'])  # Raw for calibration
        if game['result'] > 0:
            actual_wins.append(1)  # Home win
        elif game['result'] < 0:
            actual_wins.append(0)  # Away win
        else:
            actual_wins.append(0.5)  # Tie
    proj_points = np.array(proj_points)
    actual_points = np.array(actual_points)
    proj_win_probs_raw = np.array(proj_win_probs_raw)
    actual_wins = np.array(actual_wins)
    # Calibrate win probs with isotonic
    iso_calibrator = IsotonicRegression(out_of_bounds='clip').fit(proj_win_probs_raw.reshape(-1, 1), actual_wins)
    proj_win_probs = iso_calibrator.predict(proj_win_probs_raw.reshape(-1, 1))
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
    return results, iso_calibrator

# Example call (add to bottom of script)
if __name__ == "__main__":
    backtest_results, iso_calibrator = backtest_model(train_years=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], test_year=2024)  # Updated train years
    print(backtest_results)
    # Example usage with calibrator
    # simulate_matchup('KC', 'SF', team1_home=True, roof='outdoors', temp=60, wind=10, iso_calibrator=iso_calibrator)
    # simulate_matchup('DAL', 'KC', team1_home=True, roof='indoors', temp=60, wind=0, iso_calibrator=iso_calibrator)