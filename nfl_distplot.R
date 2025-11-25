library(nflfastR)
library(tidyverse)
library(nflplotR)
library(ggplot2)
library(extrafont)
library(ggrepel)
library(ggimage)
library(ggridges)
library(ggtext)
library(ggfx)
library(geomtextpath)
library(cropcircles)
library(magick)
library(glue)
library(gt)
library(gtExtras)

library(nflfastR)
library(tidyverse)
library(nflplotR)
library(ggplot2)
library(extrafont)
library(ggrepel)
library(ggimage)
library(ggridges)
library(ggtext)
library(ggfx)
library(geomtextpath)
library(cropcircles)
library(magick)
library(glue)
library(gt)
library(gtExtras)
library(nflverse)
library(xgboost)
library(xtable)

library(stringr)
library(tidyverse)
library(nnet)
library(mgcv)
library(texreg)
library(aod)
library(xtable)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(tidyverse)
library(gt)
library(gsisdecoder)

################# data prep ###########################

info <- nflfastR::teams_colors_logos
nfc <- info %>%
  filter(team_conf == "NFC")
afc <- info %>%
  filter(team_conf == "AFC")
### creating function to add SD, mean, and mean + sd
meansd <- function(x, ...) {
  mean <- mean(x)
  sd <- sd(x)
  c(mean - sd, mean, mean + sd)
}


pbp <- load_pbp(seasons=c(2024))
pbp <- add_qb_epa(pbp)
factor(pbp$posteam)
colnames(pbp)

#elo<-read_csv("https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv")


players<-calculate_player_stats(pbp=pbp)
quarterbacks<-players[which(players$position == "QB"),'player_id']

qb_map <-  pbp %>%
  filter(season_type == "REG" & down %in% c(1, 2, 3, 4) & punt_attempt == 0 
         & passer_player_id %in% quarterbacks$player_id 
         & season %in% c(2024)) %>%
  group_by(passer_player_id,posteam_type) %>% 
  summarise(
    QB_name = max(passer_player_name,na.rm = TRUE),
    QB_EPA = median(qb_epa,na.rm = TRUE),
    pass_ypa =  sum(passing_yards,na.rm = TRUE)/sum(pass_attempt,na.rm = TRUE)
    
  )

#qb_map$QB_EPA<-scale(qb_map$QB_EPA)
# qb_map$pass_ypa<-scale(qb_map$pass_ypa)



data <- pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4), punt_attempt == 0) %>%
  group_by(posteam, play_id )%>%
  summarise(
    OFF_EPA = mean(epa, na.rm = TRUE),
    QB_EPA = mean(qb_epa,na.rm = TRUE),
    QB = max(passer_player_name,na.rm = TRUE),
    fumble = sum(fumble_lost),
    interception = sum(interception),
    pass_ypa = sum(passing_yards,na.rm = TRUE)/sum(pass_attempt,na.rm = TRUE),
    rush_ypa = sum(rushing_yards,na.rm = TRUE)/sum(rush_attempt,na.rm = TRUE),
    pass_yds = sum(passing_yards,na.rm = TRUE),
    rush_yds = sum(rushing_yards,na.rm = TRUE),
    ypp = sum(sum(rushing_yards,na.rm = TRUE) + sum(passing_yards,na.rm = TRUE))/n(),
    drives = max(drive,na.rm = TRUE)/2,
    #    home_points = max(home_score,na.rm = TRUE),
    #    away_points = max(away_score,na.rm = TRUE),
    home_team = max(home_team,na.rm = TRUE),
    pressures_allowed = (sum(sack,na.rm = TRUE)+sum(qb_hit,na.rm = TRUE) ), 
    total_plays_off = n(),
    points = if_else(max(posteam,na.rm = TRUE) == max(home_team,na.rm = TRUE), max(home_score,na.rm = TRUE),max(away_score,na.rm = TRUE))
    
  )


#----------------------------- NFC EPA -------------------------- #
ggplot(data = data %>% filter(posteam %in% nfc$team_abbr), aes(x = OFF_EPA,
                                                            y = reorder(posteam,
                                                                        OFF_EPA),
                                                            fill = posteam)) +
  geom_density_ridges(quantile_lines = T,
                      quantile_fun = meansd,
                      color = "#d0d0d0") +
  nflplotR::scale_fill_nfl() +
  xlim(-5,5) + 
  #scale_x_continuous(breaks = scales::pretty_breaks()) +
  
  #nfl_analytics_theme() +
  xlab("EPA") +
  ylab("") +
  labs(title = "Distribution of EPA/play",
       subtitle = "NFC",
       caption = "Created By Harrison Eller
       Copyright: Statletics") +
  #xlim(-5,5)
  theme(axis.text.y = element_nfl_logo(size = 1.25)) +
  theme(plot.title = element_text(family = "Trebuchet MS", face="bold", size=20, hjust=0)) +
  theme(plot.subtitle = element_text(family = "Trebuchet MS", face="bold", size=12, hjust=0)) 



#----------------------------- AFC EPA -------------------------- #
ggplot(data = data %>% filter(posteam %in% afc$team_abbr), aes(x = OFF_EPA,
                                                               y = reorder(posteam,
                                                                           OFF_EPA),
                                                               fill = posteam)) +
  geom_density_ridges(quantile_lines = T,
                      quantile_fun = meansd,
                      color = "#d0d0d0") +
  nflplotR::scale_fill_nfl() +
  xlim(-5,5) + 
  #scale_x_continuous(breaks = scales::pretty_breaks()) +
  #nfl_analytics_theme() +
  xlab("EPA") +
  ylab("") +
  labs(title = "Distribution of EPA/play",
       subtitle = "AFC",
       caption = "Created By Harrison Eller
       Copyright: Statletics") +
  theme(axis.text.y = element_nfl_logo(size = 1.25)) +
  theme(plot.title = element_text(family = "Trebuchet MS", face="bold", size=20, hjust=0)) +
  theme(plot.subtitle = element_text(family = "Trebuchet MS", face="bold", size=12, hjust=0)) 




#----------------------------- NFC EPA/allowed -------------------------- #
ggplot(data = data %>% filter(defteam %in% nfc$team_abbr), aes(x = epa,
                                                               y = reorder(defteam,
                                                                           -epa),
                                                               fill = defteam)) +
  geom_density_ridges(quantile_lines = T,
                      quantile_fun = meansd,
                      color = "#d0d0d0") +
  nflplotR::scale_fill_nfl() +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  #nfl_analytics_theme() +
  xlab("EPA") +
  ylab("") +
  labs(title = "Distribution of EPA/play Allowed",
       subtitle = "NFC",
       caption = "Created By Harrison Eller
       Copyright: Statletics") +
  theme(axis.text.y = element_nfl_logo(size = 1.25)) +
  theme(plot.title = element_text(family = "Trebuchet MS", face="bold", size=20, hjust=0)) +
  theme(plot.subtitle = element_text(family = "Trebuchet MS", face="bold", size=12, hjust=0)) 



#----------------------------- AFC EPA/allowed -------------------------- #
ggplot(data = data %>% filter(defteam %in% afc$team_abbr), aes(x = epa,
                                                               y = reorder(defteam,
                                                                           -epa),
                                                               fill = defteam)) +
  geom_density_ridges(quantile_lines = T,
                      quantile_fun = meansd,
                      color = "#d0d0d0") +
  nflplotR::scale_fill_nfl() +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  #nfl_analytics_theme() +
  xlab("EPA") +
  ylab("") +
  labs(title = "Distribution of EPA/play Allowed",
       subtitle = "AFC",
       caption = "Created By Harrison Eller
       Copyright: Statletics") +
  theme(axis.text.y = element_nfl_logo(size = 1.25)) +
  theme(plot.title = element_text(family = "Trebuchet MS", face="bold", size=20, hjust=0)) +
  theme(plot.subtitle = element_text(family = "Trebuchet MS", face="bold", size=12, hjust=0)) 