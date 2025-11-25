import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, LpStatus

def optimize_showdown_lineup(
    csv_file='DK DAL-LV Showdown Fantasy and Ownership Projections.csv',
    home_team=None,
    away_team=None,
    locked_players=[],
    excluded_players=[],  # Players to exclude from the lineup
    locked_captain=None,  # New: Player to lock as captain (None for no lock)
    min_total_own=100.0,
    max_total_own=120.0,
    ownership_col='Total Own',  # Adjusted: 'Total Own' for flex
    salary_cap=50000,
    use_ownership_constraint=True,
    min_value_threshold=0.0,  # Optional: filter players with Value below this
    value_weight=5.0,  # Weight to prioritize higher Value in objective (0 to disable)
    num_lineups=1,  # Number of near-optimal lineups to generate
    max_players_per_team=None  # Maximum number of players from the same team (None for no limit)
):
    """
    Optimizes a DraftKings NFL DFS Showdown Captain Mode lineup to maximize projected points under salary cap,
    with a bias towards higher value players via weighted objective.
    
    - Filters players to those from home_team and away_team.
    - Lineup: 1 Captain (1.5x points/salary) + 5 Flex.
    - Generates multiple near-optimal lineups by iteratively solving and excluding previous selections.
    - Objective: Maximize sum(Proj * (1 + 0.5 * is_captain) + value_weight * Value)
    - Supports locking in and excluding players (locked as selected, not necessarily captain).
    - Supports locking a specific player as captain.
    - Optional total ownership constraint (sum of ownership % for the lineup).
    - Filters players by minimum value if specified.
    
    Returns: List of lineups, each a list of selected players with their details (captain first).
    """
    # Check for conflicts between locked and excluded
    conflicts = set(locked_players) & set(excluded_players)
    if conflicts:
        raise ValueError(f"Players cannot be both locked and excluded: {conflicts}")
    
    if locked_captain and locked_captain in excluded_players:
        raise ValueError(f"Player '{locked_captain}' cannot be locked as captain and excluded.")
    
    if not home_team or not away_team:
        raise ValueError("Must specify home_team and away_team for Showdown mode.")
    
    # Read and preprocess the CSV
    df = pd.read_csv(csv_file)
    
    # Filter to players from the two teams
    df = df[df['Team'].isin([home_team, away_team])]
    
    if df.empty:
        raise ValueError(f"No players found for teams {home_team} and {away_team}.")
    
    # Rename columns for consistency
    df = df.rename(columns={'Name': 'Player', 'Position': 'DK Pos'})
    
    # Salaries and projections are already numeric, but ensure
    df['Salary'] = df['Salary'].astype(int)
    df['CPT Salary'] = df['CPT Salary'].astype(int)
    df['Projection'] = df['Projection'].astype(float)
    df['CPT Projection'] = df['CPT Projection'].astype(float)
    
    # Compute Value as Projection / (Salary / 1000)
    df['Value'] = df['Projection'] / (df['Salary'] / 1000)
    
    # Ownership: 'Total Own' for flex, 'CPT Own' for captain
    df['Own'] = df['Total Own'].astype(float)
    df['CPT Own'] = df['CPT Own'].astype(float)
    
    # Add Opp column since not present
    df['Opp'] = df['Team'].apply(lambda t: away_team if t == home_team else home_team)
    
    # Filter by min value if specified
    if min_value_threshold > 0:
        df = df[df['Value'] >= min_value_threshold]
    
    # Group players by position (for stacking)
    positions = {
        'QB': df[df['DK Pos'] == 'QB']['Player'].tolist(),
        'RB': df[df['DK Pos'] == 'RB']['Player'].tolist(),
        'WR': df[df['DK Pos'] == 'WR']['Player'].tolist(),
        'TE': df[df['DK Pos'] == 'TE']['Player'].tolist(),
        'DST': df[df['DK Pos'] == 'DST']['Player'].tolist(),
        'K': df[df['DK Pos'] == 'K']['Player'].tolist()  # Added for kickers
    }
    
    # All players
    all_players = df['Player'].tolist()
    
    # Group by team for stacking and max per team
    team_groups = df.groupby('Team')
    
    # Create PuLP problem
    prob = LpProblem("DFS_Showdown_Optimizer", LpMaximize)
    
    # Binary variables: selected and captain
    player_vars = {p: LpVariable(f"select_{p}", cat='Binary') for p in all_players}
    captain_vars = {p: LpVariable(f"captain_{p}", cat='Binary') for p in all_players}
    
    # Constraints: captain <= selected
    for p in all_players:
        prob += captain_vars[p] <= player_vars[p]
    
    # Exactly 1 captain
    prob += lpSum(captain_vars[p] for p in all_players) == 1
    
    # Total selected = 6
    prob += lpSum(player_vars[p] for p in all_players) == 6
    
    # Objective base: proj (adjusted for captain) + value weight
    objective = lpSum(
        (df[df['Player'] == p]['Projection'].values[0] * player_vars[p] +
         (df[df['Player'] == p]['CPT Projection'].values[0] - df[df['Player'] == p]['Projection'].values[0]) * captain_vars[p] +
         value_weight * df[df['Player'] == p]['Value'].values[0] * player_vars[p])
        for p in all_players
    )
    
    prob += objective
    
    # Salary constraint: base for flex, CPT for captain
    prob += lpSum(
        df[df['Player'] == p]['Salary'].values[0] * (player_vars[p] - captain_vars[p]) +
        df[df['Player'] == p]['CPT Salary'].values[0] * captain_vars[p]
        for p in all_players
    ) <= salary_cap
    
    # Optional: Max players per team
    if max_players_per_team is not None:
        for team, group in team_groups:
            players_in_team = group['Player'].tolist()
            prob += lpSum(player_vars[p] for p in players_in_team) <= max_players_per_team
    
    # Lock in players (as selected)
    for locked in locked_players:
        if locked not in all_players:
            raise ValueError(f"Locked player '{locked}' not found in CSV.")
        prob += player_vars[locked] == 1
    
    # Lock captain if specified
    if locked_captain:
        if locked_captain not in all_players:
            raise ValueError(f"Locked captain '{locked_captain}' not found in CSV.")
        prob += player_vars[locked_captain] == 1
        prob += captain_vars[locked_captain] == 1
    
    # Exclude players
    for excluded in excluded_players:
        if excluded not in all_players:
            raise ValueError(f"Excluded player '{excluded}' not found in CSV.")
        prob += player_vars[excluded] == 0
    
    # Optional: Total ownership constraint (adjust for captain)
    if use_ownership_constraint:
        prob += lpSum(
            df[df['Player'] == p]['Own'].values[0] * (player_vars[p] - captain_vars[p]) +
            df[df['Player'] == p]['CPT Own'].values[0] * captain_vars[p]
            for p in all_players
        ) >= min_total_own
        prob += lpSum(
            df[df['Player'] == p]['Own'].values[0] * (player_vars[p] - captain_vars[p]) +
            df[df['Player'] == p]['CPT Own'].values[0] * captain_vars[p]
            for p in all_players
        ) <= max_total_own
    
    # Generate multiple lineups
    lineups = []
    for lineup_num in range(num_lineups):
        # Solve the problem
        status = prob.solve()
        
        if status != 1:  # Optimal solution not found
            print(f"Optimization status for lineup {lineup_num + 1}: {LpStatus[status]}")
            break
        
        # Extract selected players
        selected = []
        selected_players = []  # For exclusion
        captain = None
        for p in all_players:
            if value(player_vars[p]) == 1:
                row = df[df['Player'] == p].iloc[0]
                is_captain = value(captain_vars[p]) == 1
                selected.append({
                    'Player': p,
                    'Position': row['DK Pos'],
                    'Team': row['Team'],
                    'Salary': row['CPT Salary'] if is_captain else row['Salary'],
                    'Proj': row['CPT Projection'] if is_captain else row['Projection'],
                    'Own': row['CPT Own'] if is_captain else row['Own'],
                    'Value': row['Value'],  # Base value
                    'Is_Captain': is_captain
                })
                selected_players.append(p)
                if is_captain:
                    captain = selected[-1]
        
        # Sort: captain first, then others
        if captain:
            selected = [captain] + [s for s in selected if not s['Is_Captain']]
        
        lineups.append(selected)
        
        # Add exclusion constraint for this lineup
        prob += lpSum(player_vars[p] for p in selected_players) <= 5  # 6-1
    
    # Print lineups nicely
    for idx, lineup in enumerate(lineups):
        print(f"\nLineup {idx + 1}:")
        df_lineup = pd.DataFrame(lineup)
        df_lineup = df_lineup[['Position', 'Player', 'Team', 'Salary', 'Proj', 'Own', 'Value', 'Is_Captain']]
        print(df_lineup.to_string(index=False))
        
        total_proj = df_lineup['Proj'].sum()
        total_salary = df_lineup['Salary'].sum()
        total_own = df_lineup['Own'].sum()
        print(f"Total Projected Points: {total_proj:.2f}")
        print(f"Total Salary: {total_salary}")
        print(f"Total Ownership: {total_own:.2f}")
    
    return lineups

# Example usage:
lineups = optimize_showdown_lineup(home_team='DAL', away_team='LV', 
                                   locked_players=["Geno Smith"], 
                                   locked_captain='Tre Tucker', 
                                   excluded_players=['Malik Davis'], 
                                   salary_cap=50000,
                                   min_total_own=100, max_total_own=160,
                                   use_ownership_constraint=True, value_weight=5.0, 
                                   num_lineups=10, max_players_per_team=5)