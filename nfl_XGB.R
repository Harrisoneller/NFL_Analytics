library(nflfastR)
library(nflplotR)
library(dplyr)
library(tidyverse)
setwd("C:/Users/harri/OneDrive/Desktop/LTB/NFL Analytics")


df <- read.csv('etn_rush.csv')

player = load_player_stats(seasons=2022)
colnames(player)

player = subset(player, player_display_name %in% df$player_display_name)
player<-player %>% 
  group_by(player_id, player_display_name) %>% 
  select(recent_team,rushing_yards,player_id,player_display_name) %>% 
  summarize(ypg = mean(rushing_yards),
            team = max(recent_team))

df['player_id'] = 0
df['team'] = "TEST"
for(i in 1:nrow(df)){
  for (j in 1:nrow(player)){
    if(player$player_display_name[j] == df$player_display_name[i]){df$player_id[i] <- player$player_id[j]}
    if(player$player_display_name[j] == df$player_display_name[i]){df$team[i] <- player$team[j]}
  }
}

dfp <- df %>% arrange(desc(rush_yards_over_expected))

dfp = dfp[1:10,] 
  
ggplot2::ggplot(dfp, aes(x = reorder(player_id, -rush_yards_over_expected), y = rush_yards_over_expected)) +
  ggplot2::geom_col(aes(color = team, fill = team), width = 0.5) +
  nflplotR::geom_nfl_headshots(aes(player_gsis = player_id), width = 0.075, vjust = 0.45) +
  nflplotR::scale_color_nfl(type = "secondary") +
  nflplotR::scale_fill_nfl(alpha = 0.4) +
  ggplot2::labs(
    title = "2022 NFL Rushing Yards Over Expected",
    y = "Rushing Yards Over Expected"
  ) +
  ylim(0,420) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold"),
    plot.title.position = "plot",
    # it's obvious what the x-axis is so we remove the title
    axis.title.x = ggplot2::element_blank(),
    # this line triggers the replacement of gsis ids with player headshots
    axis.text.x = element_nfl_headshot(size = 1)
  )