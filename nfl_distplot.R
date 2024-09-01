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

df <- pbp %>%
  filter(season_type == "REG", down %in% c(1, 2, 3, 4)) %>%
  group_by(defteam,game_id) %>% 
  summarize(
    OFF_YDS = mean(epa, na.rm = TRUE)
  )

### plotting the data
ggplot(data = df %>% filter(defteam %in% nfc$team_abbr), aes(x = OFF_YDS,
                                                            y = reorder(defteam,
                                                            -OFF_YDS),
                                                            fill = defteam)) +
  geom_density_ridges(quantile_lines = T,
                      quantile_fun = meansd,
                      color = "#d0d0d0") +
  nflplotR::scale_fill_nfl() +
  scale_x_continuous(breaks = scales::pretty_breaks()) +
  #nfl_analytics_theme() +
  xlab("Average EPA/play Allowed") +
  ylab("") +
  labs(title = "Distribution of EPA/play Allowed",
       subtitle = "NFC",
       caption = "Created By Harrison Eller
       Copyright: Statletics") +
  theme(axis.text.y = element_nfl_logo(size = 1.25)) +
  theme(plot.title = element_text(family = "Trebuchet MS", face="bold", size=20, hjust=0)) +
  theme(plot.subtitle = element_text(family = "Trebuchet MS", face="bold", size=12, hjust=0)) 