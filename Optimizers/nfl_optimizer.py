import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, LpStatus

def optimize_dfs_lineup(
    csv_file='DraftKings NFL DFS Projections -- Main Slate.csv',
    locked_players=[],
    excluded_players=[],  # Players to exclude from the lineup
    min_total_own=100.0,
    max_total_own=120.0,
    ownership_col='Large Field',  # or 'Small Field'
    salary_cap=50000,
    use_ownership_constraint=True,
    min_value_threshold=0.0,  # Optional: filter players with DK Value below this
    value_weight=5.0,  # Weight to prioritize higher DK Value in objective (0 to disable)
    stack_bonus=5.0,  # Bonus per stacked player (WR/TE/RB) with QB from same team (0 to disable)
    bring_back_bonus=2.0,  # Bonus per bring-back player (WR/TE/RB) from opponent team when stacked (0 to disable)
    num_lineups=1,  # Number of near-optimal lineups to generate
    max_players_per_team=None  # New: Maximum number of players from the same team (None for no limit)
):
    """
    Optimizes a DraftKings NFL DFS lineup to maximize projected points under salary cap,
    with a bias towards higher value players via weighted objective.
    
    - Adds a bonus to encourage stacking QB with 1-2 teammates (WR/TE/RB).
    - Adds a smaller bonus to encourage bring-back players from opponent when stacked.
    - Generates multiple near-optimal lineups by iteratively solving and excluding previous selections.
    - Objective: Maximize sum(Proj + value_weight * Value) + stack_bonus * num_stacks + bring_back_bonus * num_bring_backs
    - Supports locking in and excluding players.
    - Optional total ownership constraint (sum of ownership % for the lineup).
    - Filters players by minimum value if specified.
    - Lineup: 1 QB, 2-3 RB, 3-4 WR, 1-2 TE, 1 DST, total 9 players.
    
    Returns: List of lineups, each a list of selected players with their details.
    """
    # Check for conflicts between locked and excluded
    conflicts = set(locked_players) & set(excluded_players)
    if conflicts:
        raise ValueError(f"Players cannot be both locked and excluded: {conflicts}")
    
    # Read and preprocess the CSV
    df = pd.read_csv(csv_file)
    
    # Clean salary: remove $ and ,
    df['Salary'] = df['DK Salary'].str.replace('$', '').str.replace(',', '').astype(int)
    
    # Projections and value as float
    df['Proj'] = df['DK Proj'].astype(float)
    df['Value'] = df['DK Value'].astype(float)
    
    # Ownership: strip % and convert to float
    if ownership_col in df.columns:
        df['Own'] = df[ownership_col].str.strip('%').astype(float)
    else:
        raise ValueError(f"Ownership column '{ownership_col}' not found in CSV.")
    
    # Filter by min value if specified
    if min_value_threshold > 0:
        df = df[df['Value'] >= min_value_threshold]
    
    # Group players by position
    positions = {
        'QB': df[df['DK Pos'] == 'QB']['Player'].tolist(),
        'RB': df[df['DK Pos'] == 'RB']['Player'].tolist(),
        'WR': df[df['DK Pos'] == 'WR']['Player'].tolist(),
        'TE': df[df['DK Pos'] == 'TE']['Player'].tolist(),
        'DST': df[df['DK Pos'] == 'DST']['Player'].tolist()
    }
    
    # All players
    all_players = df['Player'].tolist()
    
    # Group by team for stacking
    team_groups = df.groupby('Team')
    stack_pairs = []
    for team, group in team_groups:
        qbs = group[group['DK Pos'] == 'QB']['Player'].tolist()
        stackables = group[group['DK Pos'].isin(['RB', 'WR', 'TE'])]['Player'].tolist()
        for qb in qbs:
            for stackable in stackables:
                stack_pairs.append((qb, stackable))
    
    # Bring-back pairs
    bring_back_pairs = []
    for qb in positions['QB']:
        row = df[df['Player'] == qb].iloc[0]
        opp_team = row['Opp']
        opp_stackables = df[(df['Team'] == opp_team) & df['DK Pos'].isin(['WR', 'TE', 'RB'])]['Player'].tolist()
        for opp_player in opp_stackables:
            bring_back_pairs.append((qb, opp_player))
    
    # Create PuLP problem
    prob = LpProblem("DFS_Lineup_Optimizer", LpMaximize)
    
    # Binary variables for each player
    player_vars = {player: LpVariable(player, cat='Binary') for player in all_players}
    
    # Objective base: total projected points + weighted value
    objective = lpSum(
        (df[df['Player'] == p]['Proj'].values[0] + value_weight * df[df['Player'] == p]['Value'].values[0]) * player_vars[p]
        for p in all_players
    )
    
    # Stacking bonuses
    z_vars = {}
    if stack_bonus > 0 and stack_pairs:
        for qb, stackable in stack_pairs:
            z_name = f"z_{qb}_{stackable}".replace(' ', '_').replace("'", '')
            z_vars[(qb, stackable)] = LpVariable(z_name, cat='Binary')
            prob += z_vars[(qb, stackable)] <= player_vars[qb]
            prob += z_vars[(qb, stackable)] <= player_vars[stackable]
        objective += stack_bonus * lpSum(z_vars.values())
    
    # Bring-back bonuses (conditional on stacking)
    y_vars = {}
    has_stack = {}
    if bring_back_bonus > 0 and bring_back_pairs:
        has_stack = {qb: LpVariable(f"has_stack_{qb.replace(' ', '_').replace("'", "")}", cat='Binary') for qb in positions['QB']}
        
        # Group stackables per QB
        qb_stack_dict = {qb: [] for qb in positions['QB']}
        for qb, stackable in stack_pairs:
            qb_stack_dict[qb].append(stackable)
        
        # Set up has_stack indicators
        for qb in positions['QB']:
            stackables = qb_stack_dict[qb]
            if stackables:
                # has_stack <= sum_z (ensures has=0 if sum=0)
                prob += has_stack[qb] <= lpSum(z_vars[(qb, s)] for s in stackables)
                # for each z: has_stack >= z (ensures has=1 if any z=1)
                for s in stackables:
                    prob += has_stack[qb] >= z_vars[(qb, s)]
        
        # Bring-back vars
        for qb, opp_player in bring_back_pairs:
            y_name = f"y_{qb}_{opp_player}".replace(' ', '_').replace("'", '')
            y_vars[(qb, opp_player)] = LpVariable(y_name, cat='Binary')
            prob += y_vars[(qb, opp_player)] <= player_vars[qb]
            prob += y_vars[(qb, opp_player)] <= player_vars[opp_player]
            prob += y_vars[(qb, opp_player)] <= has_stack[qb]
        objective += bring_back_bonus * lpSum(y_vars.values())
    
    prob += objective
    
    # Constraint: Salary cap
    prob += lpSum(df[df['Player'] == p]['Salary'].values[0] * player_vars[p] for p in all_players) <= salary_cap
    
    # Position constraints
    # Exactly 1 QB
    prob += lpSum(player_vars[p] for p in positions['QB']) == 1
    
    # Exactly 1 DST
    prob += lpSum(player_vars[p] for p in positions['DST']) == 1
    
    # RBs: 2-3 (including possible FLEX)
    prob += lpSum(player_vars[p] for p in positions['RB']) >= 2
    prob += lpSum(player_vars[p] for p in positions['RB']) <= 3
    
    # WRs: 3-4
    prob += lpSum(player_vars[p] for p in positions['WR']) >= 3
    prob += lpSum(player_vars[p] for p in positions['WR']) <= 4
    
    # TEs: 1-2
    prob += lpSum(player_vars[p] for p in positions['TE']) >= 1
    prob += lpSum(player_vars[p] for p in positions['TE']) <= 2
    
    # Total players: Exactly 9
    prob += lpSum(player_vars[p] for p in all_players) == 9
    
    # New: Limit players per team
    if max_players_per_team is not None:
        for team, group in team_groups:
            players_in_team = group['Player'].tolist()
            prob += lpSum(player_vars[p] for p in players_in_team) <= max_players_per_team
    
    # Lock in players
    for locked in locked_players:
        if locked not in all_players:
            raise ValueError(f"Locked player '{locked}' not found in CSV.")
        prob += player_vars[locked] == 1
    
    # Exclude players
    for excluded in excluded_players:
        if excluded not in all_players:
            raise ValueError(f"Excluded player '{excluded}' not found in CSV.")
        prob += player_vars[excluded] == 0
    
    # Optional: Total ownership constraint
    if use_ownership_constraint:
        prob += lpSum(df[df['Player'] == p]['Own'].values[0] * player_vars[p] for p in all_players) >= min_total_own
        prob += lpSum(df[df['Player'] == p]['Own'].values[0] * player_vars[p] for p in all_players) <= max_total_own
    
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
        for p in all_players:
            if value(player_vars[p]) == 1:
                row = df[df['Player'] == p].iloc[0]
                selected.append({
                    'Player': p,
                    'Position': row['DK Pos'],
                    'Team': row['Team'],
                    'Salary': row['Salary'],
                    'Proj': row['Proj'],
                    'Own': row['Own'],
                    'Value': row['Value']
                })
                selected_players.append(p)
        
        lineups.append(selected)
        
        # Add exclusion constraint for this lineup
        prob += lpSum(player_vars[p] for p in selected_players) <= len(selected_players) - 1
    
    # Print lineups nicely
    pos_order = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'DST': 4}
    for idx, lineup in enumerate(lineups):
        print(f"\nLineup {idx + 1}:")
        df_lineup = pd.DataFrame(lineup)
        df_lineup['pos_order'] = df_lineup['Position'].map(pos_order)
        df_lineup = df_lineup.sort_values('pos_order').drop('pos_order', axis=1)
        df_lineup = df_lineup[['Position', 'Player', 'Team', 'Salary', 'Proj', 'Own', 'Value']]
        print(df_lineup.to_string(index=False))
        
        total_proj = df_lineup['Proj'].sum()
        total_salary = df_lineup['Salary'].sum()
        total_own = df_lineup['Own'].sum()
        print(f"Total Projected Points: {total_proj:.2f}")
        print(f"Total Salary: {total_salary}")
        print(f"Total Ownership: {total_own:.2f}")
    
    return lineups

# Example usage:
lineups = optimize_dfs_lineup(
    locked_players=['Brock Purdy','Christian McCaffrey'],
    excluded_players=['J.J. McCarthy', 'Olamide Zaccheaus','Jacoby Brissett','Aaron Rodgers','Michael Penix Jr.',
    'Jakobi Meyers','Michael Wilson'],
    min_total_own=50, max_total_own=140, use_ownership_constraint=True,
    value_weight=5.0, stack_bonus=5.0, bring_back_bonus=1.5, num_lineups=10, max_players_per_team=3)
# lineups is a list of list of dicts