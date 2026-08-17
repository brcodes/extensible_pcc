require(tidyverse)

### split strings in lexicon
SPLIT_LEX <- function(lex){
  # lex has to be a vector of character strings

  # create a placeholder for split strings based on number of words in the
  # lexicon and maximum word length
  n_item <- length(lex)
  max_n_phon <- max(nchar(lex), na.rm = TRUE)
  lex_split <- matrix(nrow = n_item,
                      ncol = max_n_phon,
                      data = NA)

  # name columns with incremental numbers indicating position number
  colnames(lex_split) <- sprintf("P%02d", 1:max_n_phon)
  lex_split <- data.frame(lex_split)

  # list
  lex_split_list <- do.call(strsplit, list(lex, split = ''))

  # fill in split strings into the placeholder
  for (i in seq(1, nrow(lex_split))) {
    lex_split_item <- unlist(lex_split_list[i])
    lex_split[i, 1:length(lex_split_item)] <- lex_split_item
  }

  # return lex_split as a data.frame
  return(lex_split)
}

### positional segment frequency
Pfreq <- function(trg, lex_split){
  # trg has to be a character string
  # lex_split has to be an output of the SPLIT_LEX function

  # split up target word
  trg_split <- strsplit(trg, split = '')[[1]]

  # create a placeholder for phoneme probability at each position
  max_n_phon <- ncol(lex_split)
  Pfreq_position <- rep(NA, max_n_phon)

  # calculate position-specific phoneme probabilities
  for (i in seq(1, length(trg_split)))
  {
    Pfreq_position[i] <- mean(lex_split[,i] == trg_split[i], na.rm = TRUE)
  }

  # sum and average phoneme probability of the target word
  Pfreq_sum <- sum(Pfreq_position, na.rm = TRUE)
  Pfreq_mean <- mean(Pfreq_position, na.rm = TRUE)

  # combine positional Pfreq and average Pfreq into one vector, and label as such
  Pfreq_all <- c(Pfreq_position, Pfreq_sum, Pfreq_mean)
  names(Pfreq_all) <- c(sprintf("Pfreq%02d", seq(1, max_n_phon)),
                        "Pfreq_sum", "Pfreq_mean")

  # return a named vector of all posistional Pfreq, sum Pfreq, and average Pfreq
  return(Pfreq_all)

}

### positional biphone frequency
Bfreq <- function(trg, lex_split){
  # trg has to be a character string
  # lex_split has to be an output of the SPLIT_LEX function

  # split up target word
  trg_split <- strsplit(trg, split = '')[[1]]

  # create a placeholder for biphone frequency & strings of biphones
  max_n_phon <- ncol(lex_split)
  Bfreq_position <- rep(NA, max_n_phon - 1)
  B_position <- rep(NA, max_n_phon - 1)
  names(B_position) <- sprintf("B%02d%02d",
          seq(1, max_n_phon - 1),
          seq(2, max_n_phon))

  # calculate position-specific biphone probabilities, excluding 1-phoneme words
  if (!is.na(trg) & nchar(trg) > 1) {
    for (i in seq(1, length(trg_split) - 1))
    {
      B1 <- lex_split[, i] == trg_split[i]
      B2 <- lex_split[, i + 1] == trg_split[i + 1]
      B12 <- B1 & B2
      B12[is.na(B1 | B2)] <- NA
      Bfreq_position[i] <- mean(B12, na.rm = TRUE)
      B_position[i] <- paste(trg_split[i], trg_split[i + 1], sep = "")
    }
  }

  # sum and average biphone probability of the target word
  Bfreq_sum <- sum(Bfreq_position, na.rm = TRUE)
  Bfreq_mean <- mean(Bfreq_position, na.rm = TRUE)

  # combine positional Bfreq and average Bfreq into one vector, and label as such
  Bfreq_all <- c(Bfreq_position, Bfreq_sum, Bfreq_mean)
  names(Bfreq_all) <- c(sprintf("Bfreq%02d%02d",
                             seq(1, max_n_phon - 1),
                             seq(2, max_n_phon)),
                        "Bfreq_sum", "Bfreq_mean")

  Bfreq_all <- cbind(data.frame(t(B_position),
                                stringsAsFactors = FALSE),
                     data.frame(t(Bfreq_all)))

  # return a named vector of all posistional Bfreq, sum Bfreq, and average Bfreq
  return(Bfreq_all)
}

### neighborhood density
ND <- function(trg, lex){
  # trg has to be a character string
  # lex has to be a vector of character strings

  # create a matrix containing all possible regular expressions for neighbors of
  # a given target word, i.e., one substitution, deletion, or addition at every
  # position
  trg_len <- nchar(trg)

  if (!is.na(trg_len)) {

    regex_mtx <- data.frame(
      pattern = c(sprintf("^(.{%d}).{1}(.*)$", 0:(trg_len - 1)),
                  sprintf("^(.{%d}).{1}(.*)$", 0:(trg_len - 1)),
                  sprintf("^(.{%d})(.*)$", 0:trg_len)),
      replacement = c(rep('^\\1.{1}\\2$', trg_len),
                      rep('^\\1.{0}\\2$', trg_len),
                      rep('^\\1.{1}\\2$', trg_len + 1)),
      N_type = c(rep("substitution", trg_len),
                 rep("deletion", trg_len),
                 rep("addition", trg_len + 1)),
      N_loc = c(seq(1, trg_len),
                seq(1, trg_len),
                seq(0.5, trg_len + 0.5, 1)),
      # N_loc indicates the position at which the neighbor differs from the target
      # Note: for addition, the position number ranges from 0.5 to trg_len + 0.5,
      # with an increment of 1. This position number adjustment reflects the fact
      # that additional characters occur either before/after the target word or in
      # between two adjacent characters
      stringsAsFactors = FALSE
    )

    # generate the actual regex used to later determine whether a given word is a
    # neighbor of the target word
    regex_mtx$regex <- mapply(sub,
                              pattern = regex_mtx$pattern,
                              replacement = regex_mtx$replacement,
                              x = trg)

    # create a dataframe containing all items in the lexicon
    lex_df <- data.frame(item = lex,
                         is.neighbor = rep(FALSE, length(lex)),
                         N_type = rep(NA, length(lex)),
                         N_loc = rep(NA, length(lex)),
                         stringsAsFactors = FALSE)

    # restrict search to possible word lengths
    lex_df_sub <- subset(lex_df, nchar(item) %in% c(trg_len - 1, trg_len, trg_len + 1))

    # check restricted lexicon items against every target-specific regex
    # mark all that match as neighbors, along with their N_type and N_loc
    for (i in seq(1, length(regex_mtx$regex))) {
      N_i <- grep(regex_mtx$regex[i], lex_df_sub$item)
      lex_df_sub$is.neighbor[N_i] <- TRUE
      lex_df_sub$N_type[N_i] <- regex_mtx$N_type[i]
      lex_df_sub$N_loc[N_i] <- regex_mtx$N_loc[i]
    }

    # mark target word as self
    if (trg %in% lex) {
      lex_df_sub[lex_df_sub$item == trg, ]$N_type <- "self"
      lex_df_sub[lex_df_sub$item == trg, ]$N_loc <- NA
    }

    # list all neighbors
    neighbors <- paste(lex_df_sub[lex_df_sub$is.neighbor, ]$item, collapse = " ")

    # number of neighbors of the target word
    N_sum <- sum(lex_df_sub$is.neighbor)
    # proportion of words that are neighbors of the target word in the lexicon
    N_prop <- N_sum/nrow(lex_df)

    # descriptive stats for N_loc distribution
    # N_loc distribution indicates where the neighbors tend to cluster
    # N_loc_mean and N_loc_sd are also normalized by target word length to compare
    # across target words
    if (N_sum != 0) {
      N_loc_mean <- mean(lex_df_sub$N_loc, na.rm = TRUE)
    } else {
      N_loc_mean <- NA
    }
    N_loc_sd <- sd(lex_df_sub$N_loc, na.rm = TRUE)
    N_loc_mean_norm <- N_loc_mean/trg_len
    N_loc_sd_norm <- N_loc_sd/trg_len
  } else {
    N_sum <- NA
    N_prop <- NA
    N_loc_mean <- NA
    N_loc_sd <- NA
    N_loc_mean_norm <- NA
    N_loc_sd_norm <- NA
    neighbors <- NA
  }

  # combine ND properties in a named character vector
  N_all <- c("N_sum" = N_sum,
             "N_prop" = N_prop,
             "N_loc_mean" = N_loc_mean,
             "N_loc_sd" = N_loc_sd,
             "N_loc_mean_norm" = N_loc_mean_norm,
             "N_loc_sd_norm" = N_loc_sd_norm)

  N_all <- cbind(data.frame(t(N_all)),
                 data.frame(neighbors = neighbors,
                            stringsAsFactors = FALSE))

  return(N_all)
}

### conditional transitional probablity
CondTP <- function(trg, lex){
  # trg has to be a character string
  # lex has to be a vector of character strings

  # pad all words with spaces before onset and after offset
  trg_pad <- str_c(" ", trg, " ")
  lex_pad <- str_c(" ", lex, " ")

  # create a placeholder for transitional probability at each position
  # including before word onset and between every biphone
  max_n_phon <- max(nchar(lex), na.rm = TRUE)
  CondTP_position <- rep(NA, max_n_phon)

  # calculate transitional probability at each position
  if (!is.na(trg)) {
    for (i in seq(1, nchar(trg))) {
      lex_pre <- substr(lex_pad, 1, i)
      trg_pre <- substr(trg_pad, 1, i)
      count_pre <- sum(lex_pre == trg_pre, na.rm = TRUE)

      lex_post <- substr(lex_pad, 1, i + 1)
      trg_post <- substr(trg_pad, 1, i + 1)
      count_post <- sum(lex_post == trg_post, na.rm = TRUE)

      CondTP_position[i] <- count_post / count_pre
    }
  }

    names(CondTP_position) <- sprintf("CondTP%02d", seq(1, max_n_phon))
    return(CondTP_position)
}

### TODO
### label types of neighbors (onset, medial, rhyme)
### make Levenstein Distance function
