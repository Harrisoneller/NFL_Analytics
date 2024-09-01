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


pbp <- load_pbp(2023)
pbp <- add_qb_epa(pbp)
factor(pbp$posteam)
colnames(pbp)

players<-calculate_player_stats(pbp=pbp)
quarterbacks<-players[which(players$position == "QB"),'player_id']

qb_map <-  pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4), punt_attempt == 0, passer_player_id %in% quarterbacks$player_id) %>%
  group_by(passer_player_id,posteam_type) %>% 
  summarise(
    QB_name = max(passer_player_name,na.rm = TRUE),
    QB_EPA = median(qb_epa,na.rm = TRUE),
    pass_ypa =  sum(passing_yards,na.rm = TRUE)/sum(pass_attempt,na.rm = TRUE)
    
  )

# qb_map$QB_EPA<-scale(qb_map$QB_EPA)
# qb_map$pass_ypa<-scale(qb_map$pass_ypa)



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
  
  
  ######## code #########
  
  
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
  
  # home_df['turnovers']<-home_df$interception + home_df$fumble
  # home_df['turnovers_forced']<-home_df$interceptions_forced + home_df$fumbles_forced
  # away_df['turnovers']<-away_df$interception + away_df$fumble
  # away_df['turnovers_forced']<-away_df$interceptions_forced + away_df$fumbles_forced
  # 
  df['turnovers']<-df$interception + df$fumble
  df['turnovers_forced']<-df$interceptions_forced + df$fumbles_forced

  # scale_vars<-c('ypp', 'turnovers', 'pass_ypa', 'rush_ypa',
  #         'OFF_EPA', 'pressures_allowed' ,'OFF_EPA_allowed' ,'drives',
  #         'turnovers_forced', 'pass_ypa_allowed', 'rush_ypa_allowed','pressures_forced' )
  # scale_vars<-c('OFF_EPA','OFF_EPA_allowed')
  # for (var in scale_vars){
  #   df[var]<-scale(df[var])
  # }
  # 

  
  
  
  df$weights <- 1#ifelse(as.numeric(substr(as.character(home$season),4,4))==2,as.numeric(substr(as.character(home$season),4,4)),as.numeric(substr(as.character(home$season),4,4))*2)
  
  vars<-c('ypp', 'turnovers', 'pass_ypa', 'rush_ypa',
          'OFF_EPA', 'pressures_allowed' ,'OFF_EPA_allowed' ,'drives',
          'turnovers_forced', 'pass_ypa_allowed', 'rush_ypa_allowed','pressures_forced','posteam' )
  
  X = df[,names(df) %in% vars]
  Y = df[,'points']
  r = seq(1,nrow(X), 1)
  s = sample(r, round(length(r)*.20) ,replace = F)
  # test_sample = sample(r, round(length(r)*.30) ,replace = F)
  # val_sample = sample(test_sample, round(length(test_sample)*.1) ,replace = F)
  train_x = data.matrix(X[-s,])
  train_y = data.matrix(Y[-s,])
  test_x = data.matrix(X[s,])
  test_y = data.matrix(Y[s,])
  # train_x = data.matrix(X[-c(test_sample,val_sample),])
  # train_y = data.matrix(Y[-c(test_sample,val_sample),])
  # test_x = data.matrix(X[c(test_sample),])
  # test_y = data.matrix(Y[c(test_sample),])
  # val_x = data.matrix(X[c(val_sample),])
  # val_y = data.matrix(Y[c(val_sample),])
  # train_weights = data.matrix(data$weights[-c(test_sample,val_sample)])
  # test_weights = data.matrix(data$weights[c(test_sample)])
  # val_weights = data.matrix(data$weights[c(val_sample)])
  train_weights = data.matrix(df$weights[-s])
  test_weights = data.matrix(df$weights[s])
  
  # set up the cross-validated hyper-parameter search
  xgb_grid_1 = expand.grid(
    nrounds = 500,
    eta = c(0.01, 0.001, 0.0001),
    max_depth = c(2, 4, 6, 8, 10),
    gamma = 1
  )
  # pack the training control parameters
  xgb_trcontrol_1 = trainControl(
    method = "cv",
    number = 8,
    verboseIter = TRUE,
    returnData = FALSE,
    returnResamp = "all",                                                        # save losses across all models
    classProbs = FALSE,                                                           # set to TRUE for AUC to be computed
    summaryFunction = defaultSummary,
    allowParallel = TRUE
  )
  
  
  #define predictor and response variables in testing se
  #define final training and testing sets
  xgb_train = xgb.DMatrix(data = train_x, label = train_y, weight=train_weights)
  xgb_test = xgb.DMatrix(data = test_x, label = test_y,weight = test_weights)
  
  
  watchlist = list(train=xgb_train, test=xgb_test)
  
  params <- list(booster = "gblinear",
                 objective = "reg:absoluteerror")
  params <- list(booster = "gbtree",
                 objective = "reg:absoluteerror")
  
  xgb_base <- 0
  xgb_base <- xgb.train(params = params,
                        data = xgb_train,
                        nrounds =1000,
                        print_every_n = 100,
                        max_depth = 6,
                        eval_metric = "mae",
                        early_stopping_rounds = 150,
                        trControl = xgb_trcontrol_1,
                        tuneGrid = xgb_grid_1,
                        watchlist = watchlist)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  season=2024
  week=1
  j=5
  
  

  
  PROJECTIONS <- function(week,season,df,mix_vegas=FALSE,custom_input = FALSE,home_id=FALSE,away_id=FALSE, home_qb_id=FALSE,away_qb_id=FALSE){
  
    if(custom_input == FALSE){
    
    
  schedule <- load_schedules(seasons = season)
  schedule <- schedule[which(schedule$week == week),]
  schedule['away_total']<-schedule$total_line/2 - schedule$spread_line/2
  schedule['home_total']<-schedule$total_line/2 + schedule$spread_line/2
  projections<-tibble()

  for (j in 1:nrow(schedule)){
 
  ########### initial params ############  
  home_id = schedule$home_team[j]
  away_id = schedule$away_team[j]
  
  home_df <- df %>% 
    filter((posteam == home_id | defteam == home_id) & home_team == home_id)
  away_df <- df %>% 
    filter((posteam == away_id | defteam == away_id) & away_team == away_id)
  
  QBH = qb_map$QB_name[which(qb_map$passer_player_id == schedule$home_qb_id[j] & qb_map$posteam_type == 'home' )]
  QBA = qb_map$QB_name[which(qb_map$passer_player_id == schedule$away_qb_id[j]  & qb_map$posteam_type == 'away')]
  qb_home_input = ifelse(length(QBH) != 0, pull(qb_map[which(qb_map$QB_name == QBH & qb_map$posteam_type == 'home' ),'QB_EPA']), -0.2) #Home QB EPA 
  qb_away_input = ifelse(length(QBA) != 0, pull(qb_map[which(qb_map$QB_name == QBA & qb_map$posteam_type == 'away'),'QB_EPA']), -0.2) #Away QB EPA
  qb_ypp_home = ifelse(length(QBH) != 0, pull(qb_map[which(qb_map$QB_name == QBH & qb_map$posteam_type == 'home'),'pass_ypa']), mean(qb_map$pass_ypa) - (sd(qb_map$pass_ypa)/3)) #Away QB EPA
  qb_ypp_away = ifelse(length(QBA) != 0, pull(qb_map[which(qb_map$QB_name == QBA & qb_map$posteam_type == 'away'),'pass_ypa']), mean(qb_map$pass_ypa) - (sd(qb_map$pass_ypa)/3)) #Away QB EPA
  
  
  new_home_df <- tibble(
                        posteam = home_id,
                        OFF_EPA =  (qb_home_input + median(away_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                        pass_ypa  = ( (qb_ypp_home) + median(away_df$pass_ypa_allowed,na.rm = TRUE))/2,
                        rush_ypa =  ( median(home_df$rush_ypa,na.rm = TRUE) + median(away_df$rush_ypa_allowed,na.rm = TRUE)) /2,
                        ypp =  ( median(home_df$ypp,na.rm = TRUE) + median(away_df$ypp_allowed,na.rm = TRUE)) /2,
                        drives = ( median(home_df$drives,na.rm = TRUE) + median(away_df$drives_allowed,na.rm = TRUE)) /2,
                        pressures_allowed = ( median(home_df$pressures_allowed,na.rm = TRUE) + median(away_df$pressures_forced,na.rm = TRUE)) /2,
                        OFF_EPA_allowed = ( qb_away_input + median(home_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                        pass_ypa_allowed = ( median(home_df$pass_ypa_allowed,na.rm = TRUE) + median(away_df$pass_ypa,na.rm = TRUE)) /2,
                        rush_ypa_allowed = ( median(home_df$rush_ypa_allowed,na.rm = TRUE) + median(away_df$rush_ypa,na.rm = TRUE)) /2,
                        pressures_forced = ( median(away_df$pressures_allowed,na.rm = TRUE) + median(home_df$pressures_forced,na.rm = TRUE)) /2,
                        turnovers = ( median(home_df$turnovers,na.rm = TRUE) + median(away_df$turnovers_forced,na.rm = TRUE)) /2,
                        turnovers_forced = ( median(home_df$turnovers_forced,na.rm = TRUE) + median(away_df$turnovers,na.rm = TRUE)) /2
                        
  )
  
  
  
  
  new_away_df <- tibble(
                        posteam=away_id,
                        OFF_EPA =  ( qb_away_input+ median(home_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                        pass_ypa  = ( (qb_ypp_away)+median(home_df$pass_ypa_allowed,na.rm = TRUE))/2,
                        rush_ypa =  ( median(away_df$rush_ypa,na.rm = TRUE) + median(home_df$rush_ypa_allowed,na.rm = TRUE)) /2,
                        ypp =  ( median(away_df$ypp,na.rm = TRUE) + median(home_df$ypp_allowed,na.rm = TRUE)) /2,
                        drives = ( median(away_df$drives,na.rm = TRUE) + median(home_df$drives_allowed,na.rm = TRUE)) /2,
                        pressures_allowed = ( median(away_df$pressures_allowed,na.rm = TRUE) + median(home_df$pressures_forced,na.rm = TRUE)) /2,
                        OFF_EPA_allowed = ( qb_home_input + median(away_df$OFF_EPA_allowed,na.rm = TRUE))/2,
                        pass_ypa_allowed = ( median(away_df$pass_ypa_allowed,na.rm = TRUE) + median(home_df$pass_ypa,na.rm = TRUE)) /2,
                        rush_ypa_allowed = ( median(away_df$rush_ypa_allowed,na.rm = TRUE) + median(home_df$rush_ypa,na.rm = TRUE)) /2,
                        pressures_forced = ( median(home_df$pressures_allowed,na.rm = TRUE) + median(away_df$pressures_forced,na.rm = TRUE)) /2,
                        turnovers = ( median(away_df$turnovers,na.rm = TRUE) + median(home_df$turnovers_forced,na.rm = TRUE)) /2,
                        turnovers_forced = ( median(away_df$turnovers_forced,na.rm = TRUE) + median(home_df$turnovers,na.rm = TRUE)) /2
                       
  )
  
  new_data_home = data.matrix(new_home_df[,!names(new_home_df) %in% c("points","weights")])
  new_data_away = data.matrix(new_away_df[,!names(new_away_df) %in% c("points","weights")])
  
  predict(xgb_base,new_data_home)
  predict(xgb_base,new_data_away)
  
  new_away_df
  new_home_df

  sched <- schedule %>% 
    filter(home_team==home_id & away_team==away_id)
  
  if (mix_vegas == TRUE){
  y<- tibble( home = home_id,
              home_score = (predict(xgb_base,new_data_home) + sched$home_total[1])/2 ,
              away = away_id,
              away_score = (predict(xgb_base,new_data_away) + sched$away_total[1])/2 ,
              home_spread = -(home_score - away_score),
              total = (home_score + away_score))
  }else{
  y<- tibble( home = home_id,
              home_score = (predict(xgb_base,new_data_home)) ,
              away = away_id,
              away_score = (predict(xgb_base,new_data_away)) ,
              home_spread = -(home_score - away_score),
              total = (home_score + away_score))
  }
  
  projections<-rbind(projections,y)            
  


  
  }

    } #end non-custom portion
    
    else{
      
     
      
      home_df <- df %>% 
        filter((posteam == home_id | defteam == home_id) & home_team == home_id)
      away_df <- df %>% 
        filter((posteam == away_id | defteam == away_id) & away_team == away_id)
      
      QBH = qb_map$QB_name[which(qb_map$passer_player_id == schedule$home_qb_id & qb_map$posteam_type == 'home' )]
      QBA = qb_map$QB_name[which(qb_map$passer_player_id == schedule$away_qb_id  & qb_map$posteam_type == 'away')]
      qb_home_input = ifelse(length(QBH) != 0, pull(qb_map[which(qb_map$QB_name == QBH & qb_map$posteam_type == 'home' ),'QB_EPA']), -0.2) #Home QB EPA 
      qb_away_input = ifelse(length(QBH) != 0, pull(qb_map[which(qb_map$QB_name == QBA & qb_map$posteam_type == 'away'),'QB_EPA']), -0.2) #Away QB EPA
      
      new_home_df <- tibble(
        posteam = home_id,
        OFF_EPA =  (qb_home_input + median(away_df$OFF_EPA_allowed,na.rm = TRUE))/2,
        pass_ypa  = ( pull(qb_map[which(qb_map$QB_name == QBH & qb_map$posteam_type == 'home'),'pass_ypa'])+median(away_df$pass_ypa_allowed,na.rm = TRUE))/2,
        rush_ypa =  ( median(home_df$rush_ypa,na.rm = TRUE) + median(away_df$rush_ypa_allowed,na.rm = TRUE)) /2,
        ypp =  ( median(home_df$ypp,na.rm = TRUE) + median(away_df$ypp_allowed,na.rm = TRUE)) /2,
        drives = ( median(home_df$drives,na.rm = TRUE) + median(away_df$drives_allowed,na.rm = TRUE)) /2,
        pressures_allowed = ( median(home_df$pressures_allowed,na.rm = TRUE) + median(away_df$pressures_forced,na.rm = TRUE)) /2,
        OFF_EPA_allowed = ( qb_away_input + median(home_df$OFF_EPA_allowed,na.rm = TRUE))/2,
        pass_ypa_allowed = ( median(home_df$pass_ypa_allowed,na.rm = TRUE) + median(away_df$pass_ypa,na.rm = TRUE)) /2,
        rush_ypa_allowed = ( median(home_df$rush_ypa_allowed,na.rm = TRUE) + median(away_df$rush_ypa,na.rm = TRUE)) /2,
        pressures_forced = ( median(away_df$pressures_allowed,na.rm = TRUE) + median(home_df$pressures_forced,na.rm = TRUE)) /2,
        turnovers = ( median(home_df$turnovers,na.rm = TRUE) + median(away_df$turnovers_forced,na.rm = TRUE)) /2,
        turnovers_forced = ( median(home_df$turnovers_forced,na.rm = TRUE) + median(away_df$turnovers,na.rm = TRUE)) /2
        
      )
      
      
      
      
      new_away_df <- tibble(
        posteam=away_id,
        OFF_EPA =  ( qb_away_input+ median(home_df$OFF_EPA_allowed,na.rm = TRUE))/2,
        pass_ypa  = ( pull(qb_map[which(qb_map$QB_name == QBA & qb_map$posteam_type == 'away'),'pass_ypa'])+median(home_df$pass_ypa_allowed,na.rm = TRUE))/2,
        rush_ypa =  ( median(away_df$rush_ypa,na.rm = TRUE) + median(home_df$rush_ypa_allowed,na.rm = TRUE)) /2,
        ypp =  ( median(away_df$ypp,na.rm = TRUE) + median(home_df$ypp_allowed,na.rm = TRUE)) /2,
        drives = ( median(away_df$drives,na.rm = TRUE) + median(home_df$drives_allowed,na.rm = TRUE)) /2,
        pressures_allowed = ( median(away_df$pressures_allowed,na.rm = TRUE) + median(home_df$pressures_forced,na.rm = TRUE)) /2,
        OFF_EPA_allowed = ( qb_home_input + median(away_df$OFF_EPA_allowed,na.rm = TRUE))/2,
        pass_ypa_allowed = ( median(away_df$pass_ypa_allowed,na.rm = TRUE) + median(home_df$pass_ypa,na.rm = TRUE)) /2,
        rush_ypa_allowed = ( median(away_df$rush_ypa_allowed,na.rm = TRUE) + median(home_df$rush_ypa,na.rm = TRUE)) /2,
        pressures_forced = ( median(home_df$pressures_allowed,na.rm = TRUE) + median(away_df$pressures_forced,na.rm = TRUE)) /2,
        turnovers = ( median(away_df$turnovers,na.rm = TRUE) + median(home_df$turnovers_forced,na.rm = TRUE)) /2,
        turnovers_forced = ( median(away_df$turnovers_forced,na.rm = TRUE) + median(home_df$turnovers,na.rm = TRUE)) /2
        
      )
      
      new_data_home = data.matrix(new_home_df[,!names(new_home_df) %in% c("points","weights")])
      new_data_away = data.matrix(new_away_df[,!names(new_away_df) %in% c("points","weights")])
      
      predict(xgb_base,new_data_home)
      predict(xgb_base,new_data_away)
      
      new_away_df
      new_home_df
      

      y<- tibble( home = home_id,
                  home_score = predict(xgb_base,new_data_home),
                  away = away_id,
                  away_score = predict(xgb_base,new_data_away),
                  home_spread = -(home_score - away_score),
                  total = (home_score + away_score))
      
      projections<-rbind(projections,y)     
      

      
    }
    return(projections)
  }


  
  PROJECTIONS(week=1,season=2024,df, mix_vegas = TRUE)
  
