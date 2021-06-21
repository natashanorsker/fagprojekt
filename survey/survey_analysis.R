library(ggplot2)
library(rstudioapi)

# Getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
print( getwd() )

data <- read.csv("crop_survey_results.csv", header = T)


# mean ranks and friedman test for all query images
mean(data$score[data$model == "d"])
mean(data$score[data$model == "a"])
mean(data$score[data$model == "r"])

friedman.test(y = data$score, groups = data$model, blocks = data$person)

pairwise.wilcox.test(data$score, data$model, p.adj = "bonf")


# mean ranks and friedman test for catalog images
catalog <- split(data, data$wild)$F

mean(catalog$score[catalog$model == "d"])
mean(catalog$score[catalog$model == "a"])
mean(catalog$score[catalog$model == "r"])

friedman.test(y = catalog$score, groups = catalog$model, blocks = catalog$person)

pairwise.wilcox.test(catalog$score, catalog$model, p.adj = "bonf")


# mean ranks and friedman test for wild images
wild <- split(data, data$wild)$T

mean(wild$score[wild$model == "d"])
mean(wild$score[wild$model == "a"])
mean(wild$score[wild$model == "r"])


friedman.test(y = wild$score, groups = wild$model, blocks = wild$person)

pairwise.wilcox.test(wild$score, wild$model, p.adj = "bonf")















