
library(ggplot2)


setwd("/home/antonjorg/Downloads")

data <- read.csv("answers.csv", header = T)

colnames <- names(data)

for (i in 1:length(colnames)) {
  name <- colnames[i]
  print(substr(name, 2, 3))
}
