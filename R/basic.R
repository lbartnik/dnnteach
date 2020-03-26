#' @export
f1 <- function(x) x^2

sigma <- function(x) 1/(1+exp(-x))

dsigma_dx <- function(x) {
  s <- sigma(x)
  s*(1-s)
}

nn_response <- function(X, w, b) {
  apply(sigma(outer(X,w) + outer(rep.int(1, length(X)), b)), 1, sum)
}


#' Performs a single foward+backward step.
#'
#' Computes the response of the NN, all necessary derivaties and
#' updates parameters `w` and `b` according to the learning rate `eta`.
#'
#' @param x Scalar input.
#' @param y Expected scalar output.
#' @param w Weights of NN.
#' @param b Biases of NN.
#' @param eta Learning rate.
#'
single_step <- function(x, y, w, b, eta) {
  yhat <- a(x, w, b)

  L <- (yhat-y)^2
  dLda <- 2*(yhat-y)

  dsgm <- dsigma_dx(x*w + b)
  dadw <- dsgm * x
  dadb <- dsgm

  dLdw <- dLda * dadw
  dLdb <- dLda * dadb

  w <- w - eta * dLdw
  b <- b - eta * dLdb

  list(w = w, b = b, loss = L)
}

#' Performs an epoch.
#'
#' Performs an epoch by running `single_step` for each pair from `X` and `Y`
#'
#' @param X All inputs in the training set.
#' @param Y All expected outputs in the training set.
#' @param w Weights of NN.
#' @param b Biases of NN.
#' @param eta Learning rate.
#'
epoch <- function(X, Y, w, b, eta) {
  ans <- lapply(seq(length(X)), function(i) {
    ans <- single_step(X[i], Y[i], w, b, eta)
    w <<- ans$w
    b <<- ans$b
    ans
  })
  tail(ans, 1)
}


#' Runs gradient descent for a number of epochs.
#'
#' @param X All inputs in the training set.
#' @param Y All expected outputs in the training set.
#' @param w Weights of NN.
#' @param b Biases of NN.
#' @param eta Learning rate.
#' @param epochs Number of epochs to run gradient descent for.
#' @return `tibble` with weights, biases and their respective loss at the end of each epoch.
#'
#' @export
#' @importFrom tibble as_tibble
optimize <- function(X, Y, w, b, eta, epochs) {
  tr <- list()

  for(e in seq(epochs)) {
    et <- epoch(X, Y, w, b, eta)
    w <- last(et)$w
    b <- last(et)$b

    tr <- append(tr, et)
  }

  tr <- lapply(tr, function(p) {
    unlist(p[c("w", "b", "loss")])
  })

  as_tibble(do.call(rbind, tr))
}


#' @importFrom dplyr mutate
#' @importFrom tidyselect starts_with
#' @importFrom tidyr pivot_longer
#' @importFrom ggplot2 ggplot aes geom_line
#' @importFrom tibble tibble
#' @importFrom gridExtra grid.arrange
#'
#' @export
experiment <- function(X, Y, D = 2, w = rnorm(D), b = rnorm(D), eta = 0.002, epochs = 2000) {
  trace <- optimize(X, Y, w, b, eta, epochs)

  p1 <- plot_response(X, Y, nn_response, epoch = trace[1,])
  p2 <- plot_response(X, Y, nn_response, tail(trace, 1))

  p3 <- trace %>%
    mutate(t = seq_along(w1)) %>%
    pivot_longer(c(starts_with("w"), starts_with("b"))) %>%
    ggplot(aes(x = t, y = value, color = name)) + geom_line()

  p4 <- trace %>%
    mutate(t = seq_along(w1)) %>%
    ggplot(aes(x = t, y = Loss)) + geom_line()

  sx <- seq(min(X), max(X), 0.05)
  sd <- lapply(seq(D), function(d) {
    w <- ans[[paste0("w", d)]]
    b <- ans[[paste0("b", d)]]
    sy <- a(sx, w, b)
    tibble(x = sx, y = sy, d = as.character(d))
  })
  sd <- do.call(rbind, sd)
  p5 <- ggplot(sd, aes(x = x, y = y, color = d)) + geom_line()

  grid.arrange(p1, p2, p3, p5, nrow = 2)
}



#' @importFrom dplyr mutate bind_rows bind_cols
#' @importFrom tibble tibble expand_grid
grid <- function(X, Y, b, eta=0.01, epochs=250) {
  run_no <- 0
  enumerator <- function() {
    run_no <<- run_no + 1
    run_no
  }

  w <- seq(-4, 4, 0.1)
  W <- expand_grid(w1 = w, w2 = w)

  W %>%
    alply(1, function(w) {
      trace <- optimize(X, Y, as.numeric(w), b, eta, epochs) %>%
        mutate(stage = 'intermediate',
               i = seq(2, length(stage)+1))
      trace$stage[nrow(trace)] <- 'finish'

      bind_rows(
        bind_cols(w, tibble(b1 = b[1], b2 = b[2], stage = 'start', i = 1)),
        trace
      ) %>%
        mutate(no = enumerator())
    }) %>%
    bind_rows
}




#' Reporting
#'
#' Provide `w` and `b` or `epoch` but not all at the same time.
#'
#' @param X Vector of inputs to NN.
#' @param w NN weights
#' @param b NN biases
#' @param epoch A row from the optimization output to extract weights and biases from.
#'
#' @rdname reporting
#' @name reporting
NULL

#' @importFrom dplyr select starts_with
extract_weights <- function(w, epoch) {
  if (!xor(missing(w), missing(epoch))) {
    abort("Provide `w` or `epoch` but not both at the same time")
  }
  if (!missing(w)) return(w)
  as.numeric(select(row, starts_with("w")))
}

#' @importFrom dplyr select starts_with
extract_biases <- function(b, epoch) {
  if (!xor(missing(b), missing(epoch))) {
    abort("Provide `b` or `epoch` but not both at the same time")
  }
  if (!missing(b)) return(b)
  as.numeric(select(row, starts_with("b")))
}

#' @rdname reporting
#' @importFrom tidyr pivot_longer
#' @importFrom tibble tibble
#' @importFrom ggplot2 ggplot geom_point aes
plot_response <- function(X, Y, rsp, w, b, epoch) {
  w <- extract_weights(w, epoch)
  b <- extract_biases(w, epoch)

  yhat <- vapply(X, function(x) rsp(x, w, b), numeric(1))

  tibble(x = X, y = Y, yhat) %>%
    pivot_longer(c("y", "yhat")) %>%
    ggplot(aes(x = x, y = value, color = name)) + geom_point()
}

#' @rdname reporting
plot_neurons <- function(X, w, b, epoch) {
  w <- extract_weights(w, epoch)
  b <- extract_biases(w, epoch)

}
