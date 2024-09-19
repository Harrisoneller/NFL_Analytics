library(nflfastR)
library(tidyverse)
library(nflplotR)
library(nflreadr)
library(tidymodels)
library(bonsai)
library(nflverse)
library(ggplot2)
library(ggtext)

nfl_analytics_theme <- function(..., base_size = 12) {
  
  theme(
    text = element_text(family = "Roboto",
                        size = base_size,
                        color = "black"),
    axis.ticks = element_blank(),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    plot.title.position = "plot",
    plot.title = element_markdown(size = 16,
                                  vjust = .02,
                                  hjust = 0.5),
    plot.subtitle = element_markdown(hjust = 0.5),
    plot.caption = element_markdown(size = 8),
    panel.grid.minor = element_blank(),
    panel.grid.major =  element_line(color = "#d0d0d0"),
    panel.background = element_rect(fill = "#f7f7f7"),
    plot.background = element_rect(fill = "#f7f7f7"),
    panel.border = element_blank(),
    legend.background = element_rect(color = "#F7F7F7"),
    legend.key = element_rect(color = "#F7F7F7"),
    legend.title = element_text(face = "bold"),
    legend.title.align = 0.5,
    strip.text = element_text(face = "bold"))
}


pbp <- nflreadr::load_pbp(2017:2023)

rush_attempts <- pbp %>%
  filter(season_type == "REG") %>%
  filter(rush_attempt == 1, qb_scramble == 0,
         qb_dropback == 0, !is.na(yards_gained))

def_ypc <- rush_attempts %>%
  filter(!is.na(defteam)) %>%
  group_by(defteam, season) %>%
  summarize(def_ypc = mean(yards_gained))

rush_attempts <- rush_attempts %>%
  left_join(def_ypc, by = c("defteam", "season"))


### load participation ###
participation <- nflreadr::load_participation(seasons = 2016:2022) %>%
  select(nflverse_game_id, play_id, possession_team, offense_formation,
         offense_personnel, defense_personnel, defenders_in_box)

rush_attempts <- rush_attempts %>%
  left_join(participation, by = c("game_id" = "nflverse_game_id",
                                  "play_id" = "play_id",
                                  "posteam" = "possession_team"))

### data selection ###

rushing_data_join <- rush_attempts %>%
  group_by(game_id, rusher, fixed_drive) %>%
  mutate(drive_rush_count = cumsum(rush_attempt)) %>%
  ungroup() %>%
  group_by(game_id, rusher) %>%
  mutate(game_rush_count = cumsum(rush_attempt)) %>%
  mutate(rush_prob = (1 - xpass) * 100,
         strat_score = rush_prob / defenders_in_box,
         wp = wp * 100) %>%
  ungroup() %>%
  mutate(red_zone = if_else(yardline_100 <= 20, 1, 0),
         fg_range = if_else(yardline_100 <= 35, 1, 0),
         two_min_drill = if_else(half_seconds_remaining <= 120, 1, 0)) %>%
  select(label = yards_gained, season, week, yardline_100,
         quarter_seconds_remaining, half_seconds_remaining,
         qtr, down, ydstogo, shotgun, no_huddle,
         ep, wp, drive_rush_count, game_rush_count,
         red_zone, fg_range, two_min_drill,
         offense_formation, offense_personnel,
         defense_personnel, defenders_in_box,
         rusher, rush_prob, def_ypc, strat_score,
         rusher_player_id, posteam, defteam) %>%
  na.omit()



### ngs ###
next_gen_stats <- load_nextgen_stats(seasons = 2016:2022,
                                     stat_type = "rushing") %>%
  filter(week > 0 & season_type == "REG") %>%
  select(season, week, player_gsis_id,
         against_eight = percent_attempts_gte_eight_defenders,
         avg_time_to_los)

rushing_data_join <- rushing_data_join %>%
  left_join(next_gen_stats,
            by = c("season", "week",
                   "rusher_player_id" = "player_gsis_id")) %>%
  na.omit()



rushing_data_join <- rushing_data_join %>%
  mutate(
    ol = str_extract(offense_personnel,
                     "(?<=\\s|^)\\d+(?=\\sOL)") %>% as.numeric(),
    rb = str_extract(offense_personnel,
                     "(?<=\\s|^)\\d+(?=\\sRB)") %>% as.numeric(),
    te = str_extract(offense_personnel,
                     "(?<=\\s|^)\\d+(?=\\sTE)") %>% as.numeric(),
    wr = str_extract(offense_personnel,
                     "(?<=\\s|^)\\d+(?=\\sWR)") %>% as.numeric()) %>%
  replace_na(list(ol = 5)) %>%
  mutate(extra_ol = if_else(ol > 5, 1, 0)) %>%
  mutate(across(ol:wr, as.factor)) %>%
  select(-ol, -offense_personnel)

rushing_data_join <- rushing_data_join %>%
  mutate(dl = str_extract(defense_personnel,
                          "(?<=\\s|^)\\d+(?=\\sDL)") %>% as.numeric(),
         lb = str_extract(defense_personnel,
                          "(?<=\\s|^)\\d+(?=\\sLB)") %>% as.numeric(),
         db = str_extract(defense_personnel,
                          "(?<=\\s|^)\\d+(?=\\sLB)") %>% as.numeric()) %>%
  mutate(across(dl:db, as.factor)) %>%
  select(-defense_personnel)


rushing_data_join <- rushing_data_join %>%
  filter(qtr < 5) %>%
  mutate(qtr = as.factor(qtr),
         down = as.factor(down),
         shotgun = as.factor(shotgun),
         no_huddle = as.factor(no_huddle),
         red_zone = as.factor(red_zone),
         fg_range = as.factor(fg_range),
         two_min_drill = as.factor(two_min_drill),
         extra_ol = as.factor(extra_ol))

rushes <- rushing_data_join %>%
  select(-season, -week, -rusher, -rusher_player_id,
         -posteam, -defteam) %>%
  mutate(across(where(is.character), as.factor))






##### model #####
set.seed(1988)

rushing_split <- initial_split(rushes)
rushing_train <- training(rushing_split)
rushing_test <- testing(rushing_split)
rushing_folds <- vfold_cv(rushing_train)


rushing_recipe <-
  recipe(formula = label ~ ., data = rushing_train) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


rushing_specs <- boost_tree(
  trees = tune(),
  tree_depth = tune(), 
  min_n = tune(),
  mtry = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  learn_rate = tune(),
  stop_iter = tune()) %>%
  set_engine("lightgbm", num_leaves = tune()) %>%
  set_mode("regression")


rushing_grid <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  finalize(mtry(), rushes),
  min_n(),
  num_leaves(),
  loss_reduction(),
  sample_size = sample_prop(),
  learn_rate(),
  stop_iter(),
  size = 100)


rushing_workflow <-
  workflow() %>%
  add_recipe(rushing_recipe) %>%
  add_model(rushing_specs)

library(lightgbm)
rushing_tune <- tune_grid(rushing_workflow,
                          resample = rushing_folds,
                          grid = rushing_grid,
                          control_grid(save_pred = TRUE))


best_params <- rushing_tune %>%
  select_best(metric = "rmse")

best_params


rushing_final_workflow <- rushing_workflow %>%
  finalize_workflow(best_params)

final_model <- rushing_final_workflow %>%
  fit(data = rushing_test)

rushing_predictions <- predict(final_model, rushing_data_join)

ryoe_projs <- cbind(rushing_data_join, rushing_predictions) %>%
  rename(actual_yards = label,
         exp_yards = .pred)


mean_ryoe <- ryoe_projs %>%
  dplyr::group_by(season) %>%
  summarize(nfl_mean_ryoe = mean(actual_yards) - mean(exp_yards))

ryoe_projs <- ryoe_projs %>%
  left_join(mean_ryoe, by = c("season" = "season"))

ryoe_projs <- ryoe_projs %>%
  mutate(ryoe = actual_yards - exp_yards + nfl_mean_ryoe)

for_plot <- ryoe_projs %>%
  group_by(rusher) %>%
  summarize(
    rushes = n(),
    team = last(posteam),
    yards = sum(actual_yards),
    exp_yards = sum(exp_yards),
    ypc = yards / rushes,
    exp_ypc = exp_yards / rushes,
    avg_ryoe = mean(ryoe)) %>%
  arrange(-avg_ryoe)

teams <- nflreadr::load_teams(current = TRUE)

for_plot <- for_plot %>%
  left_join(teams, by = c("team" = "team_abbr"))


### plot ###
library(ggpmisc)
library(ggrepel)
ggplot(data = for_plot, aes(x = yards, y = exp_yards)) +
  stat_poly_line(method = "lm", se = FALSE,
                 linetype = "dashed", color = "black") +
  stat_poly_eq(mapping = use_label(c("R2", "P")),
               p.digits = 2, label.x = .35, label.y = 3) +
  geom_point(color = for_plot$team_color2, size = for_plot$rushes / 165) +
  geom_point(color = for_plot$team_color, size = for_plot$rushes / 200) +
  scale_x_continuous(breaks = scales::pretty_breaks(),
                     labels = scales::comma_format()) +
  scale_y_continuous(breaks = scales::pretty_breaks(),
                     labels = scales::comma_format()) +
  labs(title = "**Actual Rushing Yards vs. Expected Rushing Yards**",
       subtitle = "*2016 - 2023 | Model: LightGBM*",
       caption = "*Statletics*
       **Harrison Eller**") +
  xlab("Actual Rushing Yards") +
  ylab("Expected Rushing Yards") +
  #nfl_analytics_theme() +
  geom_text_repel(data = filter(for_plot, yards >= 4600),
                  aes(label = rusher),
                  box.padding = 1.7,
                  segment.curvature = -0.1,
                  segment.ncp = 3, segment.angle = 20,
                   size = 4, fontface = "bold")







###3 more plots ####
diff_per_season <- ryoe_projs %>%
  group_by(season, rusher) %>%
  summarize(
    rushes = n(),
    team = last(posteam),
    yards = sum(actual_yards),
    exp_yards = sum(exp_yards),
    yards_diff = yards - exp_yards)

diff_per_season <- diff_per_season %>%
  left_join(teams, by = c("team" = "team_abbr"))

diff_per_season <- diff_per_season %>%
  group_by(season) %>%
  mutate(is_max_diff = ifelse(yards_diff == max(yards_diff), 1, 0))



ggplot(diff_per_season, aes(x = yards, y = exp_yards)) +
  stat_poly_line(method = "lm", se = FALSE,
                 linetype = "dashed", color = "black") +
  stat_poly_eq(mapping = use_label(c("R2", "P")),
               p.digits = 2, label.x = .20, label.y = 3) +
  geom_point(color = diff_per_season$team_color2,
             size = diff_per_season$rushes / 165) +
  geom_point(color = diff_per_season$team_color,
             size = diff_per_season$rushes / 200) +
  scale_x_continuous(breaks = scales::pretty_breaks(),
                     labels = scales::comma_format()) +
  scale_y_continuous(breaks = scales::pretty_breaks(),
                     labels = scales::comma_format()) +
  geom_label_repel(data = subset(diff_per_season,
                                 is_max_diff == 1),
                   aes(label = rusher),
                   box.padding = 1.5, nudge_y = 1, nudge_x = 2,
                   segment.curvature = -0.1,
                   segment.ncp = 3, segment.angle = 20,
                   #family = "Roboto",
                   size = 3.5,
                   fontface = "bold") +
  labs(title = "**Rushing Yards over Expected Leader Per Season**",
       subtitle = "*2016 - 2022 | Model: LightGBM*",
       caption = "*An Introduction to NFL Analytics with R*<br>
       **Brad J. Congelio**") +
  xlab("Actual Rushing Yards") +
  ylab("Expected Rushing Yards") +
  facet_wrap(~ season, scales = "free") +
  nfl_analytics_theme() +
  theme(strip.text = element_text(face = "bold", #family = "Roboto",
                                  size = 12),
        strip.background = element_rect(fill = "#F6F6F6"))




### individual player ###
j_taylor_2021 <- ryoe_projs %>%
  filter(rusher == "J.Taylor" & posteam == "IND" & season == 2021) %>%
  reframe(
    rusher = rusher,
    team = last(posteam),
    cumulative_yards = cumsum(actual_yards),
    cumulative_exyards = cumsum(exp_yards))

j_taylor_2021$cumulative_rushes = as.numeric(rownames(j_taylor_2021))

j_taylor_image <- png::readPNG("./images/j_taylor_background.png")


ggplot() +
  geom_line(aes(x = j_taylor_2021$cumulative_rushes,
                y = j_taylor_2021$cumulative_yards),
            color = "#002C5F", size = 1.75) +
  geom_line(aes(x = j_taylor_2021$cumulative_rushes,
                y = j_taylor_2021$cumulative_exyards),
            color = "#A2AAAD", size = 1.75) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 12),
                     labels = scales::comma_format()) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10),
                     labels = scales::comma_format()) +
  annotate(geom = "text", label = "Cumulative Actual Yards", x = 200, y = 1050,
           angle = 30, family = "Roboto", size = 5) +
  annotate(geom = "text", label = "Cumulative Expected Yards", x = 200, y = 700,
           angle = 30, family = "Roboto", size = 5) +
  annotation_custom(grid::rasterGrob(j_taylor_image, 
                                     width = unit(1,"npc"), 
                                     height = unit(1,"npc")),
                    175, 375, 0, 1000) +
  labs(title = "**Jonathan Taylor: 2021 Cumulative Actual Yards
       vs. Expected Yards**",
       subtitle = "*Model: **LightGBM** Using ***boost_trees()***
       in ***tidymodels****",
       caption = "*An Introduction to NFL Analytics with R*<br>
       **Brad J. Congelio**") +
  xlab("Cumulative Rushes in 2021 Season") +
  ylab("Cumulative Actual Yards and Expected Yards") +
  nfl_analytics_theme()