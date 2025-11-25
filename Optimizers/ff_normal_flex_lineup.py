import pandas as pd

# Load the projections CSV
# Columns include: Player,Pos,Team,Opp,DK Proj,FD Proj,FFPC Proj,Standard Proj,Half PPR Proj,Full PPR Proj,DK Ceiling,FD Ceiling,Slate
# We'll use "Full PPR Proj" as the projection metric (higher is better).
df = pd.read_csv('NFL Weekly Projections.csv', index_col=False)

# Clean up any extra spaces in column names
df.columns = df.columns.str.strip()

# Handle potential encoding issues (e.g., leading BOM)
if df.columns[0].startswith('\ufeff'):
    df.columns = [col.lstrip('\ufeff') for col in df.columns]

# Your team roster: list the full player names as they appear in the CSV
# Note: For DST, if it's listed as "CLE DST" in the CSV, update 'Cleveland Browns' to match exactly (e.g., 'CLE DST')
my_team = [
    'Jalen Hurts',
    'Jaxson Dart',
    'RJ Harvey',
    'Saquon Barkley',
    'Devin Neal',
    'Brashard Smith',
    'Michael Carter',
    'Keaton Mitchell',
    'Puka Nacua',
    'Jaxon Smith-Njigba',
    'Jauan Jennings',
    'Devonta Smith',
    'Luther Burden',
    'Rasheed Shahid',
    'Troy Franklin',
    'Jayden Reed',
    'Tyler Warren',
    'Tyler Loop',
]

# Filter the dataframe to only include players on your team
roster = df[df['Player'].isin(my_team)].copy()

def optimize_lineup(roster, proj_col):
    groups = {
        'QB': roster[roster['Pos'] == 'QB'].sort_values(proj_col, ascending=False).copy(),
        'RB': roster[roster['Pos'] == 'RB'].sort_values(proj_col, ascending=False).copy(),
        'WR': roster[roster['Pos'] == 'WR'].sort_values(proj_col, ascending=False).copy(),
        'TE': roster[roster['Pos'] == 'TE'].sort_values(proj_col, ascending=False).copy(),
        'K': roster[roster['Pos'] == 'K'].sort_values(proj_col, ascending=False).copy(),
        'DST': roster[roster['Pos'] == 'DST'].sort_values(proj_col, ascending=False).copy()  # Defense/Special Teams
    }

    lineup = {
        'Quarterback': None,
        'Running Back 1': None,
        'Running Back 2': None,
        'Wide Receiver 1': None,
        'Wide Receiver 2': None,
        'Tight End': None,
        'Flex': None,
        'Kicker': None,
        'Defense/Special Teams': None
    }

    # Fill mandatory positions first

    # QB
    player, groups['QB'] = get_next_best(groups['QB'])
    if player is not None:
        lineup['Quarterback'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    # RB1 and RB2
    for i in range(2):
        player, groups['RB'] = get_next_best(groups['RB'])
        if player is not None:
            lineup[f'Running Back {i+1}'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    # WR1 and WR2
    for i in range(2):
        player, groups['WR'] = get_next_best(groups['WR'])
        if player is not None:
            lineup[f'Wide Receiver {i+1}'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    # TE
    player, groups['TE'] = get_next_best(groups['TE'])
    if player is not None:
        lineup['Tight End'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    # K
    player, groups['K'] = get_next_best(groups['K'])
    if player is not None:
        lineup['Kicker'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    # DST
    player, groups['DST'] = get_next_best(groups['DST'])
    if player is not None:
        lineup['Defense/Special Teams'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    # Now fill Flex (RB/WR/TE) - best from remaining RB/WR/TE based on highest Proj
    candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
    if not candidates.empty:
        candidates.sort_values(proj_col, ascending=False, inplace=True)
        player = candidates.iloc[0]
        lineup['Flex'] = f"{player['Player']} ({player['Team']}) - Proj: {player[proj_col]}"

    return lineup

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

# Initialize lineup dictionary (without SuperFlex)
lineup = {
    'Quarterback': None,
    'Running Back 1': None,
    'Running Back 2': None,
    'Wide Receiver 1': None,
    'Wide Receiver 2': None,
    'Tight End': None,
    'Flex': None,
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

# Now fill Flex (RB/WR/TE) - best from remaining RB/WR/TE based on highest Proj
candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
if not candidates.empty:
    candidates.sort_values('Full PPR Proj', ascending=False, inplace=True)
    player = candidates.iloc[0]
    lineup['Flex'] = f"{player['Player']} ({player['Team']}) - Proj: {player['Full PPR Proj']}"

# Print the lineup
print("Optimal Starting Lineup Based on Full PPR Projections (ESPN Normal Flex):")
for position, player in lineup.items():
    print(f"{position}: {player if player else 'No player available'}")
lineup

ppr_lineup = optimize_lineup(roster, 'Full PPR Proj')

print("Optimal Starting Lineup Based on Full PPR Projections (ESPN Normal Flex):")
for position, player in ppr_lineup.items():
    print(f"{position}: {player if player else 'No player available'}")

ceiling_lineup = optimize_lineup(roster, 'DK Ceiling')

print("\nOptimal Starting Lineup Based on DK Ceiling Projections (ESPN Normal Flex):")
for position, player in ceiling_lineup.items():
    print(f"{position}: {player if player else 'No player available'}")