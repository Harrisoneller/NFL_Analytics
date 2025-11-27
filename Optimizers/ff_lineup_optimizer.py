import pandas as pd

# Load the rankings CSV (assumes it contains rankings for all positions: QB, RB, WR, TE, K, DST)
# Columns: Rank,Player Name,Team,Position,ECR,vs. ECR
# Use ECR as the ranking metric (lower is better)
df = pd.read_csv('FantasyPros-expert-rankings-4.csv', skiprows=4, index_col=False)
# Clean up any extra spaces in column names if needed
df.columns = df.columns.str.strip()

# Your team roster: list the full player names as they appear in the CSV
# Example: ['Josh Allen', 'Drake Maye', 'Christian McCaffrey', 'Saquon Barkley', 'Bijan Robinson',
#           'Ja\'Marr Chase', 'CeeDee Lamb', 'Travis Kelce', 'Brandon Aubrey', 'Baltimore Ravens']


my_team = [
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
    'Zonovan Knight',
    'Kendre Miller',
    'Jonathon Brooks',
    'Drake London',
    'Chris Olave',
    'Jaylen Waddle',
    'Deebo Samuel',
    'Brian Thomas Jr.',
    'Cedric Tillman',
    'Isaac TeSlaa',
    'Trey McBride',
    'Isaiah Likely',
    'Tyler Loop',
    'Cleveland Browns'
    # Add your players here...
]

# my_team = [
#     'Jalen Hurts',
#     'Jaxson Dart',
#     'RJ Harvey',
#     'Saquon Barkley',
#     'Devin Neal',
#     'Tank Bigsby',
#     'Puka Nacua',
#     'Jaxon Smith-Njigba',
#     'Jauan Jennings',
#     'DeVonta Smith',
#     'Luther Burden',
#     'Rasheed Shahid',
#     'Troy Franklin',
#     'Jayden Reed',
#     'Tyler Warren',
#     'Tyler Loop',
# ]

# Filter the dataframe to only include players on your team
roster = df[df['Player Name'].isin(my_team)].copy()

# Sort the entire roster by ECR ascending (for reference)
roster.sort_values('ECR', inplace=True)

# Group by position and sort each group by ECR ascending
groups = {
    'QB': roster[roster['Position'] == 'QB'].sort_values('ECR'),
    'RB': roster[roster['Position'] == 'RB'].sort_values('ECR'),
    'WR': roster[roster['Position'] == 'WR'].sort_values('ECR'),
    'TE': roster[roster['Position'] == 'TE'].sort_values('ECR'),
    'K': roster[roster['Position'] == 'K'].sort_values('ECR'),
    'DST': roster[roster['Position'] == 'DST'].sort_values('ECR')  # Assuming DST for Defense/Special Teams
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
    lineup['Quarterback'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# RB1 and RB2
for i in range(2):
    player, groups['RB'] = get_next_best(groups['RB'])
    if player is not None:
        lineup[f'Running Back {i+1}'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# WR1 and WR2
for i in range(2):
    player, groups['WR'] = get_next_best(groups['WR'])
    if player is not None:
        lineup[f'Wide Receiver {i+1}'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# TE
player, groups['TE'] = get_next_best(groups['TE'])
if player is not None:
    lineup['Tight End'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# K
player, groups['K'] = get_next_best(groups['K'])
if player is not None:
    lineup['Kicker'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# DST
player, groups['DST'] = get_next_best(groups['DST'])
if player is not None:
    lineup['Defense/Special Teams'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# Now fill SuperFlex (QB/RB/WR/TE) - prefer QB if available, else the best from RB/WR/TE based on lowest ECR
if not groups['QB'].empty:
    player, groups['QB'] = get_next_best(groups['QB'])
    lineup['SuperFlex'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"
else:
    # Find the best from remaining RB/WR/TE
    candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
    if not candidates.empty:
        candidates.sort_values('ECR', inplace=True)
        player = candidates.iloc[0]
        pos = player['Position']
        lineup['SuperFlex'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"
        # Remove from the group
        groups[pos] = groups[pos][groups[pos]['Player Name'] != player['Player Name']]

# Now fill Flex (RB/WR/TE) - best from remaining RB/WR/TE based on lowest ECR
candidates = pd.concat([groups['RB'], groups['WR'], groups['TE']])
if not candidates.empty:
    candidates.sort_values('ECR', inplace=True)
    player = candidates.iloc[0]
    lineup['Flex'] = f"{player['Player Name']} ({player['Team']}) - ECR: {player['ECR']}"

# Print the lineup
print("Optimal Starting Lineup:")
for position, player in lineup.items():
    print(f"{position}: {player if player else 'No player available'}")