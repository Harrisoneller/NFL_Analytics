import pandas as pd

def lineup_optimizer(players):
# Load the projections CSV
# Columns include: Player,Pos,Team,Opp,DK Proj,FD Proj,FFPC Proj,Standard Proj,Half PPR Proj,Full PPR Proj,DK Ceiling,FD Ceiling,Slate
# We'll use "Full PPR Proj" as the projection metric (higher is better).
    lineups= {}
    df = pd.read_csv('NFL Weekly Projections.csv', index_col=False)

    # Clean up any extra spaces in column names
    df.columns = df.columns.str.strip()

    # Handle potential encoding issues (e.g., leading BOM)
    if df.columns[0].startswith('\ufeff'):
        df.columns = [col.lstrip('\ufeff') for col in df.columns]

    # Your team roster: list the full player names as they appear in the CSV
    # Note: For DST, if it's listed as "CLE DST" in the CSV, update 'Cleveland Browns' to match exactly (e.g., 'CLE DST')
    my_team = players

    # Filter the dataframe to only include players on your team
    roster = df[df['Player'].isin(my_team)].copy()

    # Sort the entire roster by Full PPR Proj descending (for reference)
    roster.sort_values('Full PPR Proj', ascending=False, inplace=True)

    # Group by position and sort each group by Full PPR Proj descending
    groups = {
        'QB': roster[roster['Pos'] == 'QB'].sort_values('Full PPR Proj', ascending=False),
        'RB': roster[roster['Pos'] == 'RB'].sort_values('Full PPR Proj', ascending=False),
        'WR': roster[roster['Pos'] == 'WR'].sort_values('Full PPR Proj', ascending=False),
        'TE': roster[roster['Pos'] == 'TE'].sort_values('Full PPR Proj', ascending=False),
        'K': roster[roster['Pos'] == 'K'].sort_values('Full PPR Proj', ascending=False),
        'DST': roster[roster['Pos'] == 'DST'].sort_values('Full PPR Proj', ascending=False)  # Defense/Special Teams
    }

    # Function to get the next best player from a group and remove them from the group
    def get_next_best(position_group):
        if not position_group.empty:
            player = position_group.iloc[0]
            updated_group = position_group.iloc[1:]
            return player, updated_group
        return None, position_group

    # Initialize lineup dictionary
    lineup = {
        'Quarterback': None,
        'Running Back 1': None,
        'Running Back 2': None,
        'Wide Receiver 1': None,
        'Wide Receiver 2': None,
        'Tight End': None,
        'Flex': None,
        'SuperFlex': None,
        'Kicker': None,
        'Defense/Special Teams': None
    }

    # Fill mandatory positions first

    # QB
    player, groups['QB'] = get_next_best(groups['QB'])
    if player is not None:
        lineup['Quarterback'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # RB1 and RB2
    for i in range(2):
        player, groups['RB'] = get_next_best(groups['RB'])
        if player is not None:
            lineup[f'Running Back {i+1}'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # WR1 and WR2
    for i in range(2):
        player, groups['WR'] = get_next_best(groups['WR'])
        if player is not None:
            lineup[f'Wide Receiver {i+1}'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # TE
    player, groups['TE'] = get_next_best(groups['TE'])
    if player is not None:
        lineup['Tight End'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # K
    player, groups['K'] = get_next_best(groups['K'])
    if player is not None:
        lineup['Kicker'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # DST
    player, groups['DST'] = get_next_best(groups['DST'])
    if player is not None:
        lineup['Defense/Special Teams'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # Now fill SuperFlex (QB/RB/WR/TE) - prefer QB if available, else the best from RB/WR/TE based on highest Proj
    if not groups['QB'].empty:
        player, groups['QB'] = get_next_best(groups['QB'])
        lineup['SuperFlex'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"
    else:
        # Find the best from remaining RB/WR/TE
        candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
        if not candidates.empty:
            candidates.sort_values('Full PPR Proj', ascending=False, inplace=True)
            player = candidates.iloc[0]
            pos = player['Pos']
            lineup['SuperFlex'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"
            # Remove from the group
            groups[pos] = groups[pos][groups[pos]['Player'] != player['Player']]

    # Now fill Flex (RB/WR/TE) - best from remaining RB/WR/TE based on highest Proj
    candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
    if not candidates.empty:
        candidates.sort_values('Full PPR Proj', ascending=False, inplace=True)
        player = candidates.iloc[0]
        lineup['Flex'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

    # Print the lineup
    print("Optimal Starting Lineup Based on Full PPR Projections:")
    for position, player in lineup.items():
        print(f"{position}: {player if player else 'No player available'}")
        
    lineups['Full PPR Projections'] = lineup
    
    
    df = pd.read_csv('NFL Weekly Projections.csv', index_col=False)

    # Clean up any extra spaces in column names
    df.columns = df.columns.str.strip()

    # Handle potential encoding issues (e.g., leading BOM)
    if df.columns[0].startswith('\ufeff'):
        df.columns = [col.lstrip('\ufeff') for col in df.columns]

    # Your team roster: list the full player names as they appear in the CSV
    # Note: For DST, if it's listed as "CLE DST" in the CSV, update 'Cleveland Browns' to match exactly (e.g., 'CLE DST')
    my_team = players 

    # Filter the dataframe to only include players on your team
    roster = df[df['Player'].isin(my_team)].copy()

    # Sort the entire roster by DK Ceiling descending (for reference)
    roster.sort_values('DK Ceiling', ascending=False, inplace=True)

    # Group by position and sort each group by DK Ceiling descending
    groups = {
        'QB': roster[roster['Pos'] == 'QB'].sort_values('DK Ceiling', ascending=False),
        'RB': roster[roster['Pos'] == 'RB'].sort_values('DK Ceiling', ascending=False),
        'WR': roster[roster['Pos'] == 'WR'].sort_values('DK Ceiling', ascending=False),
        'TE': roster[roster['Pos'] == 'TE'].sort_values('DK Ceiling', ascending=False),
        'K': roster[roster['Pos'] == 'K'].sort_values('DK Ceiling', ascending=False),
        'DST': roster[roster['Pos'] == 'DST'].sort_values('DK Ceiling', ascending=False)  # Defense/Special Teams
    }

    # Function to get the next best player from a group and remove them from the group
    def get_next_best(position_group):
        if not position_group.empty:
            player = position_group.iloc[0]
            updated_group = position_group.iloc[1:]
            return player, updated_group
        return None, position_group

    # Initialize lineup dictionary
    lineup = {
        'Quarterback': None,
        'Running Back 1': None,
        'Running Back 2': None,
        'Wide Receiver 1': None,
        'Wide Receiver 2': None,
        'Tight End': None,
        'Flex': None,
        'SuperFlex': None,
        'Kicker': None,
        'Defense/Special Teams': None
    }

    # Fill mandatory positions first

    # QB
    player, groups['QB'] = get_next_best(groups['QB'])
    if player is not None:
        lineup['Quarterback'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # RB1 and RB2
    for i in range(2):
        player, groups['RB'] = get_next_best(groups['RB'])
        if player is not None:
            lineup[f'Running Back {i+1}'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # WR1 and WR2
    for i in range(2):
        player, groups['WR'] = get_next_best(groups['WR'])
        if player is not None:
            lineup[f'Wide Receiver {i+1}'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # TE
    player, groups['TE'] = get_next_best(groups['TE'])
    if player is not None:
        lineup['Tight End'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # K
    player, groups['K'] = get_next_best(groups['K'])
    if player is not None:
        lineup['Kicker'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # DST
    player, groups['DST'] = get_next_best(groups['DST'])
    if player is not None:
        lineup['Defense/Special Teams'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # Now fill SuperFlex (QB/RB/WR/TE) - prefer QB if available, else the best from RB/WR/TE based on highest Ceiling
    if not groups['QB'].empty:
        player, groups['QB'] = get_next_best(groups['QB'])
        lineup['SuperFlex'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"
    else:
        # Find the best from remaining RB/WR/TE
        candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
        if not candidates.empty:
            candidates.sort_values('DK Ceiling', ascending=False, inplace=True)
            player = candidates.iloc[0]
            pos = player['Pos']
            lineup['SuperFlex'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"
            # Remove from the group
            groups[pos] = groups[pos][groups[pos]['Player'] != player['Player']]

    # Now fill Flex (RB/WR/TE) - best from remaining RB/WR/TE based on highest Ceiling
    candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
    if not candidates.empty:
        candidates.sort_values('DK Ceiling', ascending=False, inplace=True)
        player = candidates.iloc[0]
        lineup['Flex'] = f"{player['Player']} ({player['Team']}) - Ceiling: {player['DK Ceiling']}"

    # Print the lineup
    print("Optimal Starting Lineup Based on DK Ceiling:")
    for position, player in lineup.items():
        print(f"{position}: {player if player else 'No player available'}")
        
    lineups['DK Ceiling'] = lineup
    
    return lineups

####----- cutthroat ----####

players = [
        'Jalen Hurts',
        'Jaxson Dart',
        'Geno Smith',
        'Jaylen Warren',
        'Rico Dowdle',
        'Bhayshul Tuten',
        'Jordan Mason',
        'Zach Charbonnet',
        'Jaydon Blue',
        'Emmanuel Wilson',
        'Kendre Miller',
        'Jonathon Brooks',
        'Drake London',
        'Chris Olave',
        'Jaylen Waddle',
        'Deebo Samuel',
        'Brian Thomas Jr.',
        'Cedric Tillman',
        'Isaac TeSlaa',
        'Dontayvion Wicks',
        'Trey McBride',
        'Isaiah Likely',
        'Tyler Loop',
        'Cleveland Browns'  # Update to exact CSV name if needed, e.g., 'CLE DST'
    ]

lineups = lineup_optimizer(players = players)
lineups


