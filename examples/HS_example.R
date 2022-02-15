##### BAYESREGSEM ILLUSTRATION: HOLZINGER & SWINEFORD DATA #####
## Author: Sara van Erp

library(dplyr)
library(ggplot2)
library(lavaan)
library(tidySEM)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

set.seed(15022022)

source("./examples/examples_fun.R")

##### 1. Generate large population data set based on original HS data -----
## compute covariance matrix original data
HSdat <- HolzingerSwineford1939
HSdatsel <- HSdat %>%
  select(paste0("x", 1:9))
obscov <- cov(HSdatsel)

## add 4th item to one factor that is uncorrelated to the other items
x10 <- rnorm(9, mean = 0, sd = 0.1)

obscovn10 <- rbind(
  cbind(obscov, x10),
  c(x10, 1)
)

## generate a large sample based on this observed covariance matrix
popdat <- simdat.fun(nobs = 5000,
                     nitem = 10,
                     covmat = obscovn10,
                     varnames = paste0("x", 1:10))

##### 2. Sample a training and test set from the population -----
## training set
trainsel <- sample(1:5000, 
                   size = 300,
                   replace = FALSE)
traindat <- popdat[trainsel, ]

## test set
testsel <- sample(1:5000,
                  size = 300,
                  replace = FALSE)
testdat <- popdat[testsel, ]

##### 3. Fit CFA on the training set and use modification indices to adapt the model -----
## Fit the model
HSmod0 <- 'visual =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed =~ x7 + x8 + x9 + x10'

fit0 <- fit.fun(mod = HSmod0,
        data = traindat)

checkfit.fun(fitind = fit0$fit)

## Adaptation 1
HSmod1 <- adapt.fun(mi = fit0$MIs,
                    mod = HSmod0)

fit1 <- fit.fun(mod = HSmod1,
                data = traindat)

checkfit.fun(fitind = fit1$fit)

## Adaptation 2
HSmod2 <- adapt.fun(mi = fit1$MIs,
                    mod = HSmod1)

fit2 <- fit.fun(mod = HSmod2,
                data = traindat)

checkfit.fun(fitind = fit2$fit)

## Adaptation 3
HSmod3 <- adapt.fun(mi = fit2$MIs,
                    mod = HSmod2)

fit3 <- fit.fun(mod = HSmod3,
                data = traindat)

checkfit.fun(fitind = fit3$fit)

## Adaptation 4
HSmod4 <- adapt.fun(mi = fit3$MIs,
                    mod = HSmod3)

fit4 <- fit.fun(mod = HSmod4,
                data = traindat)

checkfit.fun(fitind = fit4$fit)

## Adaptation 5
HSmod5 <- adapt.fun(mi = fit4$MIs,
                    mod = HSmod4)

fit5 <- fit.fun(mod = HSmod5,
                data = traindat)

checkfit.fun(fitind = fit5$fit) 

## Adaptation 6
HSmod6 <- adapt.fun(mi = fit5$MIs,
                    mod = HSmod5)

fit6 <- fit.fun(mod = HSmod6,
                data = traindat)

checkfit.fun(fitind = fit6$fit) 

## we can continue adapting, but because we only free cross-loadings the corresponding MIs are no longer very large
fit6$MIs

##### 4. Test the resulting model (with 5 adaptations) on the test set -----
fittest <- cfa(HSmod5, 
               data = testdat)

fitmeastest <- fitmeasures(fittest, 
                           c("pvalue", "cfi", "tli", "rmsea", "srmr"))
## combine all fit measures
rbind(
  fit0$fit,
  fit1$fit,
  fit2$fit,
  fit3$fit,
  fit4$fit,
  fit5$fit,
  fitmeastest
)

## Plot final model (with 5 added cross-loadings)
lay <- get_layout("", "", "visual","","textual","","","speed", "", "",
                  "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", rows = 2)
graph_sem(fittest, layout = lay)

##### 5. Fit bayesregsem model with shrinkage priors on the training set and select cross-loadings to free -----
##### 6. Compare estimates between shrinkage priors -----
##### 7. Test the resulting model on the test set -----