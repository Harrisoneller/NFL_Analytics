library(cfbfastR)
library(tidyverse)
library(lubridate)
library(dplyr)
library(readr)
#remotes::install_github(repo = "sportsdataverse/cfbfastR")
remotes::install_github(repo = "Kazink36/cfbplotR")
library(cfbplotR)
Sys.setenv(CFBD_API_KEY = "x1C/67YV6Sy98uENGd+tSvJSr82NfDxHFTmWk4QB5wGl2qxogM53QKLB5T4l6kPn")
#Sys.setenv(CFBD_API_KEY = "OMFtwopAS5WexLsewwy5BKQsUIzguwFqGz6KkjiUc6zcpKNYphzld/71fWW7pt8j")
library(stringdist)
library(tidyverse)
library(nnet)
library(mgcv)
library(texreg)
library(aod)
library(xtable)
library(xgboost)
library(readxl)
library(stringr)
library(caret)
library(car)
library(tidyverse)
library(nflplotR)
library(nflfastR)
setwd("C:/Users/harri/OneDrive/Desktop/LTB/NFL Analytics")

receiving_ngs <- read_csv("receiving_ngs.csv")
colnames(receiving_ngs)

df <- receiving_ngs %>% 
  filter(season == 2022, season_type == 'REG',week ==0) %>% 
  select( "player_display_name",                 "player_position",                     "team_abbr",                           "avg_cushion",                        
          "avg_separation",                      "avg_intended_air_yards",              "percent_share_of_intended_air_yards", "receptions",                         
          "targets",                             "catch_percentage",                    "yards",                               "rec_touchdowns",                     
          "avg_yac",                             "avg_expected_yac",                    "avg_yac_above_expectation")


TE<- df %>% 
  filter(player_position == 'TE')

WR<- df %>% 
  filter(player_position == 'WR')


rank_cols = c("avg_cushion",                        
              "avg_separation",                      "avg_intended_air_yards",              "percent_share_of_intended_air_yards", "receptions",                         
              "targets",                             "catch_percentage",                    "yards",                               "rec_touchdowns",                     
              "avg_yac",                             "avg_expected_yac",                    "avg_yac_above_expectation")

#paste(rank_cols[1],"_rank",sep="")


for (i in 1:length(rank_cols)){
  WR[paste(rank_cols[i],"_rank",sep="")] = rank(desc(WR[rank_cols[i]]),na.last = TRUE,ties.method = "max")
  TE[paste(rank_cols[i],"_rank",sep="")] = rank(desc(TE[rank_cols[i]]),na.last = TRUE,ties.method = "max")
}



