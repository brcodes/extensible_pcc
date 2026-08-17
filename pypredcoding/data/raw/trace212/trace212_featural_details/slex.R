require(tidyverse)
require(fs)
require(here)
library(foreach)
library(doParallel)
source(fs::path(here::here(), "data", "slex", "PP_ND_functions.R"))
require(Hmisc)

# WORDS
slex <- read.csv(fs::path(here::here(), "data", "slex", "slex.txt"),
                stringsAsFactors = FALSE, na.strings = "NULL",
                header = FALSE, col.names = "Phono.orig")

slex$Phono <- gsub('\\^', '_', slex$Phono.orig)

# calculating phonotactic probability and neighborhood density
slex$n_phon <- nchar(slex$Phono)
slex_lex_split <- SPLIT_LEX(slex$Phono)

## number of CPU cores used for parallel processing
n_cpu <- 10

## Pfreq
### start parallel processing
cl <- makeCluster(n_cpu)
registerDoParallel(cl)
Sys.time()

### parallel processing for loop
slex_Pfreq <- foreach(trg = slex$Phono,
                     .combine = rbind,
                     .packages = c("tidyverse", "plyr")) %dopar% {

                       trg_Pfreq <- Pfreq(trg = trg,
                                          lex_split = slex_lex_split)

                       cbind(data.frame(target = trg),
                             as.list(trg_Pfreq))
                     }

### stop parallel processing
stopCluster(cl)
Sys.time()

## Bfreq
### start parallel processing
cl <- makeCluster(n_cpu)
registerDoParallel(cl)
Sys.time()

### parallel processing for loop
slex_Bfreq <- foreach(trg = slex$Phono,
                     .combine = rbind,
                     .packages = c("tidyverse", "plyr")) %dopar% {

                       trg_Bfreq <- Bfreq(trg = trg,
                                          lex_split = slex_lex_split)

                       cbind(data.frame(target = trg),
                             as.list(trg_Bfreq))
                     }

### stop parallel processing
stopCluster(cl)
Sys.time()

## Conditional Transitional Probability
### start parallel processing
cl <- makeCluster(n_cpu)
registerDoParallel(cl)
Sys.time()

### parallel processing for loop
slex_CondTP <- foreach(trg = slex$Phono,
                      .combine = rbind,
                      .packages = c("tidyverse", "plyr")) %dopar% {

                        trg_CondTP <- CondTP(trg = trg,
                                             lex = slex$Phono)

                        cbind(data.frame(target = trg),
                              as.list(trg_CondTP))
                      }

### stop parallel processing
stopCluster(cl)
Sys.time()

## ND
### start parallel processing
cl <- makeCluster(n_cpu)
registerDoParallel(cl)
Sys.time()

### parallel processing for loop
slex_ND <- foreach(trg = slex$Phono,
                  .combine = rbind,
                  .packages = c("tidyverse", "plyr")) %dopar% {

                    trg_ND <- ND(trg = trg,
                                 lex = slex$Phono)

                    cbind(data.frame(target = trg),
                          as.list(trg_ND))
                  }

### stop parallel processing
stopCluster(cl)
Sys.time()

# combine data frames
slex_all <- cbind(slex, slex_lex_split,
                  subset(slex_Pfreq, select = -target),
                  subset(slex_Bfreq, select = -target),
                  subset(slex_CondTP, select = -target),
                  subset(slex_ND, select = -target))

# save image
write.csv(slex_all, file = "slex.csv", row.names = FALSE)
