setwd("/home/lukasz/Projects/dnnteach/work/")

library(plyr)
library(dplyr)
library(ggplot2)
library(gganimate)
library(tidyr)
library(dnnteach)


set.seed(43)

w <- rnorm(1)
b <- rnorm(1)

X <- runif(100, -2, 2)
Y <- quadratic(X) + rnorm(length(X), sd=.1)

# a single X,Y pair

single_xy <- function(X, Y, w = rnorm(1), b = rnorm(1), eta = 0.1, epochs = 50) {
  tr <- optimize(X[1], Y[1], w = w, b = b, eta = eta, epochs = epochs)
  pl <-
    tr %>%
    ddply(.(epoch), function(row) {
      bind_rows(
        tibble(x = X, y = Y, type = 'rest'),
        tibble(x = X[1], y = Y[1], type = 'train'),
        tibble(x = X, y = nn_response(X, epoch = row), type = 'neuron')
      )
    }) %>%
    as_tibble
  list(trace = tr, plot = pl)
}


all <- single_xy(X, Y, w = 3, b = 0, eta = .5)

plot_frame <- function(data) {
  ggplot() +
    geom_point(data = filter(data, type != 'neuron'), aes(x = x, y = y, color = type)) +
    geom_line(data = filter(data, type == 'neuron'), aes(x = x, y = y))
}

all$plot %>%
  filter(epoch == 1) %>%
  plot_frame

p1 <-
  all$pl %>%
  plot_frame + transition_time(epoch) + labs(title = "Epoch: {frame_time}")

anim_save("xy.gif", p1)

p2 <-
  all$trace %>%
  pivot_longer(c(w, b), "param") %>%
  ggplot() + geom_line(aes(y = epoch, x = value, color = param))

anim_save("wb.gif", p2 + transition_reveal(epoch))




runs <-
  ldply(seq(25), function(no) {
    tr <- optimize(X[1], Y[2], w = rnorm(1), b = rnorm(1), eta = 0.01, epochs = 100)
    tail(tr, 1) %>% select(-epoch)
  }) %>%
  arrange(loss)

