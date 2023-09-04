library(nflplotR)
library(tidyverse)
library(openxlsx)
setwd("C:/Users/Harrison Eller/sports-app-server")
pr <- read.csv("power_rankings.csv")

pr$AVG_EPA <- round(pr$AVG_EPA,digits = 1)
pr$AVG_EPA_Allowed <- round(pr$AVG_EPA_Allowed,digits = 1)

library(gt)
library(gtExtras)
library(nflplotR)
teams <- "https://github.com/nflverse/nflfastR-data/raw/master/teams_colors_logos.rds"
team_df <- readRDS(url(teams))

df <- pr %>% inner_join( team_df, 
                       by=c('Team'='team_abbr'))

df <- df %>% arrange(rating)


logo_table <- df %>%
  dplyr::select(Rating = rating,Logo = team_logo_espn, Team, AVG_EPA , AVG_EPA_Allowed ,Conf = team_conf) %>%
  gt() %>%
  #gtExtras::gt_img_rows(columns = team_wordmark, height = 25) %>%
  gtExtras::gt_img_rows(columns = Logo, img_source = "web", height = 30) %>%
  tab_options(data_row.padding = px(1)) %>% 
  cols_align(
    align = c('center'),
    columns = everything()
  ) %>%
  tab_header(
    title = md("***NFL Power Ratings***"),
    subtitle = md("**Statletics**")
  ) %>%
  fmt_number(
    columns = c("AVG_EPA", "AVG_EPA_Allowed")
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "bold")
    ),
    locations = cells_body(
      columns = c("AVG_EPA", "AVG_EPA_Allowed")
    )
  )












team_plot_data %>%
  transmute(Conference = conference, Home_Team = ht,
            Home_Score = round(ht_score,2),
            Away_Score = round(at_score,2), Away_Team = at, Spread_Pick = value_side) %>%
  arrange(desc(Conference)) %>%
  gt() %>%
  gt_fmt_cfb_logo(columns = c("Conference", "Spread_Pick")) %>%
  gt_fmt_cfb_wordmark(columns = c("Home_Team","Away_Team")) %>%
  cols_align(
    align = c('center'),
    columns = everything()
  ) %>%
  tab_header(
    title = md("**Scarlett Score Predictions**"),
    subtitle = md("Harrison Eller")
  ) %>%
  fmt_number(
    columns = c("Home_Score", "Away_Score")
  ) %>%
  tab_style(
    style = list(
      cell_text(weight = "bold")
    ),
    locations = cells_body(
      columns = c("Home_Score", "Away_Score")
    )
  )


