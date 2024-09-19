library(tidyverse)
library(cfbfastR)
library(cfbplotR)
library(caret)
Sys.setenv(CFBD_API_KEY = "x1C/67YV6Sy98uENGd+tSvJSr82NfDxHFTmWk4QB5wGl2qxogM53QKLB5T4l6kPn")

rec <- cfbd_recruiting_position(start_year = 2020,end_year = 2024)

## QB
qb <- rec %>% filter(position_group == "Quarterback")
process <- preProcess(as.data.frame(qb$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(qb$avg_rating))

qb$qb_scaled <- norm_scale$`qb$avg_rating`*1.5

## WR
wr <- rec %>% filter(position_group == "Receiver")
process <- preProcess(as.data.frame(wr$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(wr$avg_rating))

wr$wr_scaled <- norm_scale$`wr$avg_rating`




## WR
rb <- rec %>% filter(position_group == "Running Back")
process <- preProcess(as.data.frame(rb$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(rb$avg_rating))

rb$rb_scaled <- norm_scale$`rb$avg_rating`




## OL
ol <- rec %>% filter(position_group == "Offensive Line")
process <- preProcess(as.data.frame(ol$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(ol$avg_rating))

ol$ol_scaled <- norm_scale$`ol$avg_rating`*1.5





## DL
dl <- rec %>% filter(position_group == "Defensive Line")
process <- preProcess(as.data.frame(dl$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(dl$avg_rating))

dl$dl_scaled <- norm_scale$`dl$avg_rating`*1.5




## LB
lb <- rec %>% filter(position_group == "Linebacker")
process <- preProcess(as.data.frame(lb$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(lb$avg_rating))

lb$lb_scaled <- norm_scale$`lb$avg_rating`



## DB
db <- rec %>% filter(position_group == "Defensive Back")
process <- preProcess(as.data.frame(db$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(db$avg_rating))

db$db_scaled <- norm_scale$`db$avg_rating`



#ST

st <- rec %>% filter(position_group == "Special Teams")
process <- preProcess(as.data.frame(st$avg_rating), method=c("range"))

norm_scale <- predict(process, as.data.frame(st$avg_rating))

st$st_scaled <- norm_scale$`st$avg_rating`

names<-c("team","position group","qb_scaled","wr_scaled","rb_scaled","ol_scaled","dl_scaled","lb_scaled","db_scaled","st_scaled")
temp<-tibble(qb$team,qb$position_group,qb$qb_scaled,wr$wr_scaled,rb$rb_scaled,ol$ol_scaled,dl$dl_scaled,lb$lb_scaled,db$db_scaled,st$st_scaled)

colnames(temp)<-names


###clean

temp %>% select(team,conference,position_group,avg_rating,qb_scaled,wr_scaled,rb_scaled,ol_scaled,dl_scaled,lb_scaled,db_scaled
                ,st_scaled)

