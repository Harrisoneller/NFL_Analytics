# Load required libraries
library(shiny)
library(shinydashboard)
input_season = 2024
team_info<-cfbd_team_info(year = input_season)
stats <- cfbd_stats_season_team(2023)
TEAMS <- cfbd_stats_season_advanced(2023)
ht<-c('Tennessee')
at<-c('Alabama')

TEAMS <- inner_join(stats,TEAMS,by = 'team')
TEAMS$Total_YPG <- TEAMS$total_yds / TEAMS$games
TEAMS$Passing_YPG <- TEAMS$net_pass_yds / TEAMS$games
TEAMS$Rushing_YPG <- TEAMS$rush_yds / TEAMS$games
TEAMS$Passing_YPA <- TEAMS$net_pass_yds / TEAMS$games
TEAMS$Passing_YPC <- TEAMS$net_pass_yds / TEAMS$games
TEAMS$Rushing_YPA <- TEAMS$rush_yds / TEAMS$games
TEAMS$third_down_conv_rate<- TEAMS$third_down_convs / TEAMS$third_downs

tab <- TEAMS %>% 
  filter(team %in% c(ht,at)) %>% 
  select(team, Total_YPG, Passing_YPG,Rushing_YPG, Passing_YPA, Passing_YPC, Rushing_YPA, pass_TDs, rush_TDs, passes_intercepted, off_success_rate, off_explosiveness,off_power_success,off_passing_plays_success_rate, off_passing_plays_explosiveness, off_rushing_plays_success_rate, off_rushing_plays_explosiveness)


t <- TEAMS %>% 
  filter(team %in% c(ht,at))


gt1<-t %>%
  select(team, conference.x, pass_TDs, rush_TDs,  interceptions,fumbles_lost,Total_YPG, Passing_YPG,Rushing_YPG, Passing_YPA, Passing_YPC, Rushing_YPA, third_down_conv_rate,off_success_rate, off_explosiveness,off_power_success,off_passing_plays_success_rate, off_passing_plays_explosiveness, off_rushing_plays_success_rate, off_rushing_plays_explosiveness) %>%
  transmute(Team = team, Passing_TDs = round(pass_TDs,0), Rushing_TDs = round(rush_TDs,0),  INTs = round(interceptions,0),FUM = round(fumbles_lost,0),Total_YPG, Passing_YPG,Rushing_YPG, Passing_YPA, Passing_YPC, Rushing_YPA, third_down_conv_rate) %>% 
  gt(rowname_col = c("Team")) %>% 
  gt_fmt_cfb_logo(columns = c('Team')) %>% 
  cols_label(
    Team = md("**Team**"),
    Passing_TDs = md("**Pass TDs**"),
    Rushing_TDs = md("**Rush TDs**"),
    INTs = md("**INT**"),
    FUM = md("**FUM**"),
    Total_YPG = md("**YPG**"),
    Passing_YPG = md("**Pass YPG**"),
    Rushing_YPG = md("**Rush YPG**"),
    Passing_YPA = md("**Pass YPA**"),
    Passing_YPC = md("**Pass YPC**"),
    Rushing_YPA = md("**Rush YPA**"),
    third_down_conv_rate = md("**3D Conversion Rate**")
  ) %>% 
  fmt_number(
    columns = c(Passing_TDs,Rushing_TDs,INTs,FUM, Total_YPG, Passing_YPG,Rushing_YPG),
    decimals = 0
  ) %>% 
  fmt_number(
    columns = c(Passing_YPA, Passing_YPC, Rushing_YPA, third_down_conv_rate),
    decimals = 2
  ) %>% 
  data_color(
    columns = c(Passing_TDs),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$pass_TDs),max(TEAMS$pass_TDs)
      )
    )
  ) %>% 
  data_color(
    columns = c(Rushing_TDs),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$rush_TDs),max(TEAMS$rush_TDs)
      )
    )
  )  %>% 
  data_color(
    columns = c(INTs),
    colors = scales::col_numeric(
      palette = c('green','white','red'),
      domain = c(min(TEAMS$interceptions),max(TEAMS$interceptions)
      )
    )
  ) %>% 
  data_color(
    columns = c(FUM),
    colors = scales::col_numeric(
      palette = c('green','white','red'),
      domain = c(min(TEAMS$fumbles_lost),max(TEAMS$fumbles_lost)
      )
    )
  ) %>% 
  data_color(
    columns = c(Total_YPG),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$Total_YPG),max(TEAMS$Total_YPG)
      )
    )
  )  %>% 
  data_color(
    columns = c(Passing_YPG),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$Passing_YPG),max(TEAMS$Passing_YPG)
      )
    )
  ) %>% 
  data_color(
    columns = c(Rushing_YPG),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$Rushing_YPG),max(TEAMS$Rushing_YPG)
      )
    )
  )%>% 
  data_color(
    columns = c(Passing_YPA),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$Passing_YPA),max(TEAMS$Passing_YPA)
      )
    )
  )%>% 
  data_color(
    columns = c(Passing_YPC	),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$Passing_YPC),max(TEAMS$Passing_YPC)
      )
    )
  )%>% 
  data_color(
    columns = c(Rushing_YPA	),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$Rushing_YPA),max(TEAMS$Rushing_YPA)
      )
    )
  )%>% 
  data_color(
    columns = c(third_down_conv_rate),
    colors = scales::col_numeric(
      palette = c('red','white','green'),
      domain = c(min(TEAMS$third_down_conv_rate),max(TEAMS$third_down_conv_rate)
      )
    )
  ) %>% 
  tab_header(
    title = md("**Statistical Comparison**"),
    subtitle = md("2023 Season")
  ) %>% 
  tab_footnote(
    footnote = "YPG = Yards Per Game",
    locations = cells_column_labels(
      columns = c(Total_YPG, Passing_YPG, Rushing_YPG)
    )
  ) %>% 
  tab_footnote(
    footnote = "YPA = Yards Per Attempt",
    locations = cells_column_labels(
      columns = c(Passing_YPA, Rushing_YPA)
    )
  )  %>% 
  tab_footnote(
    footnote = "YPC = Yards Per Completion",
    locations = cells_column_labels(
      columns = c(Passing_YPC)
    )
  )




# Define UI for the dashboard
ui <- dashboardPage(
  dashboardHeader(disable = TRUE),  # Disable the header
  dashboardSidebar(disable = TRUE), # Disable the sidebar
  dashboardBody(
    # Include custom CSS for positioning images
    tags$head(
      tags$style(HTML("
        body {
          background-color: #ffffff; /* Set background color to white */
        }
        .top-left-image {
          position: absolute;
          top: 10px;
          left: 10px;
        }
        .top-right-image {
          position: absolute;
          top: 10px;
          right: 10px;
        }

        .content-wrapper {
          position: relative;
          padding-top: 30px; /* Reduced padding to move table up */
          background-color: #ffffff; /* Ensure background is white */
        }
        .centered-table-wrapper {
          display: flex;
          justify-content: center;
          align-items: center;
          height: calc(100vh - 30px); /* Reduced height to move table up */
          background-color: #ffffff; /* Ensure background is white */
        }
        .centered-table-container {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 80%; /* Adjust width as needed */
          max-width: 100%;
          overflow: auto;
          background-color: #ffffff; /* Ensure background is white */
        }
        .box {
          background-color: #ffffff; /* Set box background color to white */
          border: 1px solid #dcdcdc; /* Optional: Add a border to boxes */
        }
      "
      ))
    ),
    # Include images in the body
    tags$div(
      class = "top-left-image",
      tags$img(src = team_info$logo[which(team_info$school == ht)], height = "200px", width = "200px")
    ),
    tags$div(
      class = "top-right-image",
      tags$img(src = team_info$logo[which(team_info$school == at)], height = "200px", width = "200px")
    ),
    
    fluidRow(
      div(class = "centered-table-wrapper",
          div(class = "centered-table-container",
              withSpinner(gt_output("gt1"))
        )
      )
    )
  )
)



  # Load required libraries
  library(shiny)
  library(shinydashboard)
  library(shinycssloaders)
  library(gt)
  
  # Create a sample data frame for the gt table



  
  
  
  # Define server logic
  server <- function(input, output) {
    output$gt1 <- render_gt({
      gt1
    })
  }
  

# Run the application
shinyApp(ui = ui, server = server)
