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

### prelim ###
YPP<-function(season=2024,week){

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

pbp <- load_pbp(c(2023,2024))
pbp <- add_qb_epa(pbp)

players<-calculate_player_stats(pbp=pbp)
quarterbacks<-players[which(players$position == "QB"),'player_id']


qb_map <-  pbp %>%
  filter(season_type == "REG" & down %in% c(1, 2, 3, 4) & punt_attempt == 0 
         & passer_player_id %in% quarterbacks$player_id 
         & season %in% c(2023,2024)) %>%
  group_by(passer_player_id,posteam_type) %>% 
  summarise(
    QB_name = max(passer_player_name,na.rm = TRUE),
    QB_EPA = median(qb_epa,na.rm = TRUE),
    pass_ypa =  sum(passing_yards,na.rm = TRUE)/sum(pass_attempt,na.rm = TRUE)
    
  )



## ---------------------------- function ---------------------------- ##

NFL_YPP_SPREAD <- function(home_id, away_id, QBH, QBA) {

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

pbp <- load_pbp(c(2023,2024))
pbp <- add_qb_epa(pbp)


players<-calculate_player_stats(pbp=pbp)
quarterbacks<-players[which(players$position == "QB"),'player_id']


qb_map <-  pbp %>%
  filter(season_type == "REG" & down %in% c(1, 2, 3, 4) & punt_attempt == 0 
         & passer_player_id %in% quarterbacks$player_id 
         & season %in% c(2023,2024)) %>%
  group_by(passer_player_id,posteam_type) %>% 
  summarise(
    QB_name = max(passer_player_name,na.rm = TRUE),
    QB_EPA = median(qb_epa,na.rm = TRUE),
    pass_ypa =  sum(passing_yards,na.rm = TRUE)/sum(pass_attempt,na.rm = TRUE)
    
  )


df_off <- pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4), punt_attempt == 0) %>%
  group_by(posteam, game_id )%>%
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


df_def<- pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4), punt_attempt == 0) %>%
  group_by(defteam, game_id )%>%
  summarise(
    OFF_EPA_allowed = mean(epa, na.rm = TRUE),
    QB_EPA_allowed = mean(qb_epa,na.rm = TRUE),
    #QB = max(passer_player_name,na.rm = TRUE),
    fumbles_forced = sum(fumble_lost),
    interceptions_forced = sum(interception),
    pass_ypa_allowed = sum(passing_yards,na.rm = TRUE)/sum(pass_attempt,na.rm = TRUE),
    rush_ypa_allowed = sum(rushing_yards,na.rm = TRUE)/sum(rush_attempt,na.rm = TRUE),
    pass_yds_allowed = sum(passing_yards,na.rm = TRUE),
    rush_yds_allowed = sum(rushing_yards,na.rm = TRUE),
    ypp_allowed = sum(sum(rushing_yards,na.rm = TRUE) + sum(passing_yards,na.rm = TRUE))/n(),
    drives_allowed = max(drive,na.rm = TRUE)/2,
    #    home_points = max(home_score,na.rm = TRUE),
    #    away_points = max(away_score,na.rm = TRUE),
    away_team = max(away_team,na.rm = TRUE),
    pressures_forced = (sum(sack,na.rm = TRUE)+sum(qb_hit,na.rm = TRUE) ), 
    total_plays_def = n(),
    points_allowed = if_else(max(defteam,na.rm = TRUE) == max(home_team,na.rm = TRUE), max(away_score,na.rm = TRUE),max(home_score,na.rm = TRUE))
    
    
  )



df<-cbind(df_off,df_def)



####### INPUTS #########

# home_id = 'DET'; home_qb = 'J.Goff'
# away_id = 'LA';away_qb = 'M.Stafford'
# 

home_qb = QBH
away_qb = QBA

######## code #########

home_df <- df %>% 
  filter(posteam == home_id | defteam == home_id)
away_df <- df %>% 
  filter(posteam == away_id | defteam == away_id)


# n_train <- sample(seq(1,nrow(df),1) , .75*nrow(df))
# n_test <- seq(1,nrow(df),1) [!(seq(1,nrow(df),1) %in% n_train)]
# 
# 
# team_df <- df %>% 
#   filter(posteam == id | defteam == id)
# 
# n_train <- sample(seq(1,nrow(team_df),1) , .75*nrow(team_df))
# n_test <- seq(1,nrow(team_df),1) [!(seq(1,nrow(team_df),1) %in% n_train)]
# team_df['turnovers']<-team_df$interception + team_df$fumble
# team_df['turnovers_forced']<-team_df$interceptions_forced + team_df$fumbles_forced
# 
# train_rows = team_df[n_train,]
# test_rows = team_df[n_test,]

home_df['turnovers']<-home_df$interception + home_df$fumble
home_df['turnovers_forced']<-home_df$interceptions_forced + home_df$fumbles_forced
away_df['turnovers']<-away_df$interception + away_df$fumble
away_df['turnovers_forced']<-away_df$interceptions_forced + away_df$fumbles_forced

################################ home ##############################
off_model_home <- lm(ypp ~ turnovers + pass_ypa + rush_ypa +
                  + OFF_EPA + pressures_allowed + OFF_EPA_allowed + drives,data = home_df)

def_model_home <- lm(ypp_allowed ~  turnovers_forced + pass_ypa_allowed + rush_ypa_allowed + OFF_EPA_allowed 
                + OFF_EPA + pressures_forced + drives,data = home_df)
################################ away ##############################
off_model_away <- lm(ypp ~ turnovers + pass_ypa + rush_ypa +
                       + OFF_EPA + pressures_allowed + OFF_EPA_allowed + drives,data = away_df)

def_model_away <- lm(ypp_allowed ~  turnovers_forced + pass_ypa_allowed + rush_ypa_allowed + OFF_EPA_allowed 
                     + OFF_EPA + pressures_forced + drives,data = away_df)


summary(off_model_home);summary(def_model_home)
summary(off_model_away);summary(def_model_away)

 # qb_home_input = ifelse(QBH != FALSE, pull(qb_map[which(qb_map$QB_name == home_qb & qb_map$posteam_type == 'home'),'QB_EPA']), 0) #Home QB EPA 
 # qb_away_input = ifelse(QBA != FALSE, pull(qb_map[which(qb_map$QB_name == away_qb & qb_map$posteam_type == 'away'),'QB_EPA']), 0) #Away QB EPA
  qb_home_input = ifelse(QBH != FALSE, pull(qb_map[which(qb_map$QB_name == home_qb ),'QB_EPA']), 0) #Home QB EPA 
  qb_away_input = ifelse(QBA != FALSE, pull(qb_map[which(qb_map$QB_name == away_qb),'QB_EPA']), 0) #Away QB EPA


new_home_df <- tibble(turnovers = ( median(home_df$turnovers,na.rm = TRUE) + median(away_df$turnovers_forced,na.rm = TRUE)) /2,
                      pass_ypa  = ( pull(qb_map[which(qb_map$QB_name == home_qb),'pass_ypa'])+median(away_df$pass_ypa_allowed,na.rm = TRUE))/2,
                      rush_ypa =  ( median(home_df$rush_ypa,na.rm = TRUE) + median(away_df$rush_ypa_allowed,na.rm = TRUE)) /2,
                      OFF_EPA =  (qb_home_input + median(away_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                      pressures_allowed = ( median(home_df$pressures_allowed,na.rm = TRUE) + median(away_df$pressures_forced,na.rm = TRUE)) /2,
                      drives = ( median(home_df$drives,na.rm = TRUE) + median(away_df$drives_allowed,na.rm = TRUE)) /2,
                      
                      turnovers_forced = ( median(home_df$turnovers_forced,na.rm = TRUE) + median(away_df$turnovers,na.rm = TRUE)) /2,
                      pass_ypa_allowed = ( median(home_df$pass_ypa_allowed,na.rm = TRUE) + median(away_df$pass_ypa,na.rm = TRUE)) /2,
                      rush_ypa_allowed = ( median(home_df$rush_ypa_allowed,na.rm = TRUE) + median(away_df$rush_ypa,na.rm = TRUE)) /2,
                      OFF_EPA_allowed = ( qb_away_input + median(home_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                      pressures_forced = ( median(away_df$pressures_allowed,na.rm = TRUE) + median(home_df$pressures_forced,na.rm = TRUE)) /2,
                      )




new_away_df <- tibble(turnovers = ( median(away_df$turnovers,na.rm = TRUE) + median(home_df$turnovers_forced,na.rm = TRUE)) /2,
                      pass_ypa  = ( pull(qb_map[which(qb_map$QB_name == away_qb),'pass_ypa'])+median(home_df$pass_ypa_allowed,na.rm = TRUE))/2,
                      rush_ypa =  ( median(away_df$rush_ypa,na.rm = TRUE) + median(home_df$rush_ypa_allowed,na.rm = TRUE)) /2,
                      OFF_EPA =  ( qb_away_input+ median(home_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                      pressures_allowed = ( median(away_df$pressures_allowed,na.rm = TRUE) + median(home_df$pressures_forced,na.rm = TRUE)) /2,
                      drives = ( median(away_df$drives,na.rm = TRUE) + median(home_df$drives_allowed,na.rm = TRUE)) /2,
                      
                      turnovers_forced = ( median(away_df$turnovers_forced,na.rm = TRUE) + median(home_df$turnovers,na.rm = TRUE)) /2,
                      pass_ypa_allowed = ( median(away_df$pass_ypa_allowed,na.rm = TRUE) + median(home_df$pass_ypa,na.rm = TRUE)) /2,
                      rush_ypa_allowed = ( median(away_df$rush_ypa_allowed,na.rm = TRUE) + median(home_df$rush_ypa,na.rm = TRUE)) /2,
                      OFF_EPA_allowed = ( qb_home_input + median(away_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                      pressures_forced = ( median(home_df$pressures_allowed,na.rm = TRUE) + median(away_df$pressures_forced,na.rm = TRUE)) /2,
)



home_diff = predict(off_model_home,newdata = new_home_df)-predict(def_model_home,newdata = new_home_df)

away_diff = predict(off_model_away,newdata = new_away_df)-predict(def_model_away,newdata = new_away_df)

 home_spread <- -(home_diff - away_diff)/.2
 print(-(home_diff - away_diff)/.2);print(summary(off_model_home))
 
 y<- tibble( home = home_id,
             home_off_ypp = predict(off_model_home,newdata = new_home_df),
             home_def_ypp = predict(def_model_home,newdata = new_home_df),
             home_ypp_diff = home_diff,
             away = away_id,
             away_off_ypp = predict(off_model_away,newdata = new_away_df),
             away_def_ypp = predict(def_model_away,newdata = new_away_df),
             away_ypp_diff = away_diff
            )
 
 
return(y)
 
}






## ----------------------------  ---------------------------- ##
# season = 2024
# week = 2
schedule <- load_schedules(seasons = season)
schedule <- schedule[which(schedule$week == week),]
schedule['away_total']<-schedule$total_line/2 - schedule$spread_line/2
schedule['home_total']<-schedule$total_line/2 + schedule$spread_line/2
schedule['home_ypp']=0;schedule['away_ypp']=0

ypp_proj<-tibble()
for (game in 1:nrow(schedule)){

y<-NFL_YPP_SPREAD(home_id = schedule$home_team[game],away_id = schedule$away_team[game], 
                  QBH = qb_map$QB_name[which(qb_map$passer_player_id == schedule$home_qb_id[game])][1],
                  QBA = qb_map$QB_name[which(qb_map$passer_player_id == schedule$away_qb_id[game])][1])
ypp_proj<-rbind(ypp_proj,y)
}

return(ypp_proj)

  
}