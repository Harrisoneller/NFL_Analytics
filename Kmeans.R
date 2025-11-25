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
library(mgcv)
library(texreg)
library(aod)
library(readr)
library(stringr)
library(caret)
library(car)
library(gt)
library(nflverse)
library(factoextra)
library(cowplot)
library(gghighlight)


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
pbp<- clean_pbp(pbp)
pbp <- add_qb_epa(pbp)
factor(pbp$posteam)
colnames(pbp)


qb_map <-  pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4), punt_attempt == 0) %>%
  group_by(passer_player_id,posteam_type) %>% 
  summarise(
    QB_name = max(passer_player_name,na.rm = TRUE),
    QB_EPA = mean(qb_epa,na.rm = TRUE),
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

off <- df %>% 
  group_by(posteam) %>% 
  summarise(
    EPA = median(OFF_EPA)
  )


def <- df %>% 
  group_by(defteam) %>% 
  summarise(
    EPA_allowed = median(OFF_EPA_allowed)
  )

agg <- off %>% 
  inner_join(def, c('posteam'='defteam'))


ggplot2::ggplot(agg, aes(x = -EPA_allowed, y = EPA)) +
  #ggplot2::geom_abline(slope = -1.5, intercept = seq(0.4, -0.3, -0.1), alpha = .2) +
  nflplotR::geom_median_lines(aes(x0 = EPA_allowed , y0 = EPA)) +
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














#----------------------------------------------------- k-means -----------------------------------------------------#

############## gather team aggregate data ###############

off <- df %>% 
  group_by(posteam) %>% 
  summarise(
    EPA = median(OFF_EPA),
    ypp = median(ypp),
    pass_ypa = median(pass_ypa),
    rush_ypa = median(rush_ypa),
    turnovers = median(interception)+median(fumble)
    
  )


def <- df %>% 
  group_by(defteam) %>% 
  summarise(
    EPA_allowed = median(OFF_EPA_allowed),
    ypp_allowed = median(ypp),
    pass_ypa_allowed = median(pass_ypa),
    rush_ypa_allowed = median(rush_ypa),
    turnovers_forced = median(interceptions_forced)+median(fumbles_forced)
    
  )

agg <- off %>% 
  inner_join(def, c('posteam'='defteam'))

teams<-agg$posteam

agg <- agg %>%
  select(-posteam)

rownames(agg) <- teams



################ principle component analysis #################


teams_pca <- prcomp(agg, center = TRUE, scale = TRUE)

fviz_pca_biplot(teams_pca, geom = c("point", "text")) +
  xlim(-6, 3) +
  labs(title = "**PCA Biplot: PC1 and PC2**") +
  xlab("PC1 - 35.8%") +
  ylab("PC2 - 24.6%")

get_eigenvalue(teams_pca)


fviz_eig(teams_pca, addlabels = TRUE) +
  xlab("Principal Component") +
  ylab("% of Variance Explained") +
  labs(title = "**PCA Analysis: Scree Plot**")



pc1 <- fviz_contrib(teams_pca, choice = "var", axes = 1)
pc2 <- fviz_contrib(teams_pca, choice = "var", axes = 2)
pc3 <- fviz_contrib(teams_pca, choice = "var", axes = 3)
pc4 <- fviz_contrib(teams_pca, choice = "var", axes = 4)
pc5 <- fviz_contrib(teams_pca, choice = "var", axes = 5)

plot_grid(pc1, pc2, pc3, pc4, pc5)

################ construct K-means algo #################


k <- 5
pca_scores <- teams_pca$x
team_kmeans <- kmeans(pca_scores, centers = k)
team_kmeans$cluster


cluster_assignment <- team_kmeans$cluster
agg$cluster <- cluster_assignment


kmean_dataviz <- agg %>%
  mutate(cluster = case_when(
    cluster == 1 ~ "Cluster 1",
    cluster == 2 ~ "Cluster 2",
    cluster == 3 ~ "Cluster 3",
    cluster == 4 ~ "Cluster 4",
    cluster == 5 ~ "Cluster 5"))

kmean_data_long <- kmean_dataviz %>%
  gather("Variable", "Value", -cluster)

ggplot(kmean_data_long, aes(x = Variable, y = Value, color = cluster)) +
  geom_point(size = 3) +
  facet_wrap(~ cluster) +
  scale_color_brewer(palette = "Set1") +
  gghighlight(use_direct_label = FALSE) +
  theme(axis.text = element_text(angle = 90, size = 8),
        strip.text = element_text(face = "bold"),
        legend.position = "none")
