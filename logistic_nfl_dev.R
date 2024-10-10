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

pbp <- load_pbp()
pbp <- add_qb_epa(pbp)
factor(pbp$posteam)
colnames(pbp)


qb_map <-  pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4), punt_attempt == 0) %>%
  group_by(passer_player_id) %>% 
  summarise(
    QB_name = max(passer_player_name,na.rm = TRUE),
    QB_EPA = mean(qb_epa,na.rm = TRUE)
    
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
df[which(df$game_id...2 != df$game_id...17),]
id = 'BUF'

n_train <- sample(seq(1,nrow(df),1) , .75*nrow(df))
n_test <- seq(1,nrow(df),1) [!(seq(1,nrow(df),1) %in% n_train)]


team_df <- df %>% 
  filter(posteam == id | defteam == id)

n_train <- sample(seq(1,nrow(team_df),1) , .75*nrow(team_df))
n_test <- seq(1,nrow(team_df),1) [!(seq(1,nrow(team_df),1) %in% n_train)]
team_df['turnovers']<-team_df$interception + team_df$fumble
team_df['turnovers_forced']<-team_df$interceptions_forced + team_df$fumbles_forced

team_df['win']<-0
for (i in 1:nrow(team_df)){
  if(team_df$points[i] > team_df$points_allowed[i]){team_df$win[i]<-1}
}

train_rows = team_df[n_train,]
test_rows = team_df[n_test,]


# model <- glm(win ~ turnovers + turnovers_forced + pass_ypa + rush_ypa + pass_ypa_allowed + rush_ypa_allowed + OFF_EPA_allowed 
#               + OFF_EPA  + pressures_allowed + pressures_forced ,data = train_rows)

off_model <- lm(ypp ~ turnovers + pass_ypa + rush_ypa +
             + OFF_EPA + pressures_allowed + OFF_EPA_allowed + drives,data = train_rows)

def_model <- lm(ypp_allowed ~  turnovers_forced + pass_ypa_allowed + rush_ypa_allowed + OFF_EPA_allowed 
                + OFF_EPA + pressures_forced + drives,data = train_rows)


summary(off_model)
summary(def_model)

predict(off_model,test_rows)

test_rows
preds <- predict.glm(model, test_rows,type = 'response')
probs_test <- plogis(preds)
plot(model)


test <- df[n_test,]
test['correct']<-0

for (j in 1:nrow(df[n_test,])){
  if (test$win[j] == 1 & probs_test[j] >= .5){test$correct[j]<- 1}
}


sum(test$correct)/nrow(test)


##########################



summary(model)
plot(model)


df['win']<-0
df['turnovers']<-df$interception + df$fumble
df['turnovers_forced']<-df$interceptions_forced + df$fumbles_forced

for (i in 1:nrow(df)){
  if(df$points[i] > df$points_allowed[i]){df$win[i]<-1}
}
model <- glm(win ~ turnovers + turnovers_forced + pass_ypa + rush_ypa + pass_ypa_allowed + rush_ypa_allowed + OFF_EPA_allowed 
             + OFF_EPA  + pressures_allowed + pressures_forced + posteam,data = df[n_train,],family = c('gaussian'))


summary(model)
preds <- predict(model, df[n_test,])


preds <- predict.glm(model, df[n_test,],type = 'response')
probs_test <- plogis(preds)
plot(model)


test <- df[n_test,]
test['correct']<-0

for (j in 1:nrow(df[n_test,])){
  
  if (test$win[j] == 1 & probs_test[j] >= .5){test$correct[j]<- 1}
}


sum(test$correct)/nrow(test)
