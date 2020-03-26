create_enumerator <- function() {
  run_no <- 0

  function() {
    run_no <<- run_no + 1
    run_no
  }
}
