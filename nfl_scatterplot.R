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

info <-nflfastR::teams_colors_logos
nfc<-info %>% 
  filter(team_conf == "NFC")
afc<-info %>% 
  filter(team_conf == "AFC")

pbp<-load_pbp()

nflfastR::

df <- pbp %>% 
  filter(season_type =="REG",down %in% c(1,2,3,4)) %>% 
  group_by(defteam) %>% 
  summarize(
    EPA_Allowed = mean(epa)
  )


df$EPA_Allowed_rank <-rank(df$EPA_Allowed)


df2 <- pbp %>% 
  filter(season_type =="REG",down %in% c(1,2,3,4)) %>% 
  group_by(posteam) %>% 
  summarize(
    EPA = mean(epa)
  )

df2$EPA_rank <-rank(-df2$EPA)


epa_comp <- inner_join(df,df2,by=c("defteam"="posteam"))

df_off <- pbp %>% 
  filter(season_type =="REG",down %in% c(1,2,3,4)) %>% 
  group_by(posteam,game_id) %>% 
  summarize(
    EPA = mean(epa),
    defteam = max(defteam)
  )

epa_comp$epa_allowed_sector<-ifelse(epa_comp$EPA_Allowed_rank <=16,1,2)
df_off$opponent_sector<-0

for(i in 1:nrow(df_off)){
  df_off$opponent_sector[i]<-epa_comp$epa_allowed_sector[which(epa_comp$Team == df_off$defteam[i])]
}

df_off <-df_off %>% 
  group_by(posteam,opponent_sector) %>% 
  summarize(
  avg_epa = mean(EPA)
  )

df_off$epa_above<-0.00000000000001
df_off$epa_below<-0.00000000000001

for(i in 1:nrow(df_off)){
ifelse(df_off$opponent_sector[i] ==1, df_off$epa_above[i]<-df_off$avg_epa[i],df_off$epa_below[i]<-df_off$avg_epa[i])
}

df_off <- df_off %>%   
  group_by(posteam) %>% 
  summarize(
    avg_epa_above = mean(epa_above),
    avg_epa_below = mean(epa_below)
  )

ggplot2::ggplot(df_off, aes(x = avg_epa_below, y = avg_epa_above)) +
  #ggplot2::geom_abline(slope = -1.5, intercept = seq(0.4, -0.3, -0.1), alpha = .2) +
  nflplotR::geom_mean_lines(aes(x0 = avg_epa_below , y0 = avg_epa_above)) +
  nflplotR::geom_nfl_logos(aes(team_abbr = posteam), width = 0.065, alpha = 0.7) +
  ggplot2::labs(
    x = "Offensive EPA/play vs. Lower Half Defense",
    y = "Offensive EPA/play vs. Upper Half Defense",
    caption = "Data: @nflfastR",
    title = "2023 OFF EPA/play Against Good Defenses Versus OFF EPA/play Against Bad Defenses"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold"),
    plot.title.position = "plot"
  ) 
  #ylim(min(df_off$avg_epa_above)-0.01,min(df_off$avg_epa_above)+0.01)
  #ggplot2::scale_y_reverse()
