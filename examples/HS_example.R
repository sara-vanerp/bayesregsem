##### BAYESREGSEM ILLUSTRATION: HOLZINGER & SWINEFORD DATA #####
## Author: Sara van Erp

library(dplyr)
library(ggplot2)
library(lavaan)
library(tidySEM)
library(regsem)
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

summary(fittest, fit.measures = TRUE)

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
parsel <- c("L_main_C", "L_cross_C", "phi_C", "psi")

## Ridge 1
mod.ridge <- stan_model("./examples/HS_ridge.stan")
standat.ridge <- list(N = nrow(traindat),
                      P = ncol(traindat),
                      Q = 3,
                      C = 20,
                      y = traindat,
                      s0 = 0.03) # variance of .001
fit.ridge1 <- sampling(mod.ridge, data = standat.ridge, iter = 4000, pars = parsel)
save(fit.ridge1, file = "./examples/HS_fit_ridge1.RData")

## Regularized horseshoe 1
mod.reghs <- stan_model("./examples/HS_reghs.stan")
standat.reghs1 <- list(N = nrow(traindat),
                      P = ncol(traindat),
                      Q = 3,
                      C = 20,
                      y = traindat,
                      scale_global = 1,
                      nu_global = 1,
                      nu_local = 1,
                      slab_scale = 1,
                      slab_df = 1)
fit.reghs1 <- sampling(mod.reghs, data = standat.reghs1, iter = 4000, pars = parsel)
save(fit.reghs1, file = "./examples/HS_fit_reghs1.RData")

## Regularized horseshoe 2
standat.reghs2 <- list(N = nrow(traindat),
                      P = ncol(traindat),
                      Q = 3,
                      C = 20,
                      y = traindat,
                      scale_global = 0.01,
                      nu_global = 1,
                      nu_local = 1,
                      slab_scale = 1,
                      slab_df = 1)
fit.reghs2 <- sampling(mod.reghs, data = standat.reghs2, iter = 4000, pars = parsel)
save(fit.reghs2, file = "./examples/HS_fit_reghs2.RData")

##### 6. Compare estimates between shrinkage priors -----
load("./examples/HS_fit_ridge1.RData")
load("./examples/HS_fit_reghs1.RData")
load("./examples/HS_fit_reghs2.RData")

fitls <- list("ridge1" = fit.ridge1,
              "reghs1" = fit.reghs1,
              "reghs2" = fit.reghs2)

## create df with posterior means and CIs
estdf <- do.call(rbind, df_est(fitls, CI = "95%"))

## plot
crossF1 <- c("L_cross_C[7]", "L_cross_C[8]", "L_cross_C[9]",
             "L_cross_C[13]", "L_cross_C[14]", "L_cross_C[15]", "L_cross_C[16]")
crossF2 <- c("L_cross_C[1]", "L_cross_C[2]", "L_cross_C[3]",
             "L_cross_C[17]", "L_cross_C[18]", "L_cross_C[19]", "L_cross_C[20]")
crossF3 <- c("L_cross_C[4]", "L_cross_C[5]", "L_cross_C[6]",
             "L_cross_C[10]", "L_cross_C[11]", "L_cross_C[12]")

# select parameters to plot
plotdat <- estdf[which(estdf$par %in% crossF1), ]
plotdat <- estdf[grep("L_main_C", estdf$par), ]
plotdat <- estdf[grep("phi_C", estdf$par), ]
plotdat <- estdf[grep("psi", estdf$par), ]
pd <- position_dodge(0.3)
ggplot(plotdat, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd)

##### 7. Test the resulting model on the test set -----
## Based on the 95% CI, we would add cross-loading 15
HSmodB <- 'visual =~ x1 + x2 + x3 + x9
textual =~ x4 + x5 + x6
speed =~ x7 + x8 + x9 + x10'

fittestB <- cfa(HSmodB, 
               data = testdat)

fitmeastestB <- fitmeasures(fittestB, 
                           c("pvalue", "cfi", "tli", "rmsea", "srmr"))
rbind(
  fitmeastest,
  fitmeastestB
)

## Plot final model (with 1 added cross-loading)
lay <- get_layout("", "", "visual","","textual","","","speed", "", "",
                  "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", rows = 2)
graph_sem(fittestB, layout = lay)

##### 8. Apply classical regsem to the training set -----
## Fit model with all cross-loadings in lavaan
HS.mod.cl <- 'visual =~ lv1*x1 + lv2*x2 + lv3*x3 + lv4*x4 + lv5*x5 + lv6*x6 + lv7*x7 + lv8*x8 + lv9*x9 + lv10*x10
textual =~ lt1*x1 + lt2*x2 + lt3*x3 + lt4*x4 + lt5*x5 + lt6*x6 + lt7*x7 + lt8*x8 + lt9*x9 + lt10*x10
speed =~ ls1*x1 + ls2*x2 + ls3*x3 + ls4*x4 + ls5*x5 + ls6*x6 + ls7*x7 + ls8*x8 + ls9*x9 + ls10*x10'

fit.lav <- cfa(HS.mod.cl, data = traindat, std.lv = TRUE)

## Use the resulting fitobject in regsem
cl <- c("lv4", "lv5", "lv6", "lv7", "lv8", "lv9", "lv10",
        "lt1", "lt2", "lt3", "lt7", "lt8", "lt9", "lt10",
        "ls1", "ls2", "ls3", "ls4", "ls5", "ls6")
fit.regsem <- cv_regsem(fit.lav, 
                        pars_pen = cl,
                        n.lambda = 25,
                        jump = .025,
                        type = "lasso",
                        metric = "BIC")
save(fit.regsem, file = "./examples/HS_fit_regsem.RData")

## Get fit results
load("./examples/HS_fit_regsem.RData")
summary(fit.regsem)
round(fit.regsem$fits, 2)
plot(fit.regsem, show.minimum = "BIC")

## Get estimates
est.reg <- fit.regsem$final_pars
names(est.reg) <- c("L_main_C[1]", "L_main_C[2]", "L_main_C[3]",
                    "L_cross_C[7]", "L_cross_C[8]", "L_cross_C[9]", "L_cross_C[13]", "L_cross_C[14]", "L_cross_C[15]", "L_cross_C[16]",
                    "L_cross_C[1]", "L_cross_C[2]", "L_cross_C[3]", 
                    "L_main_C[4]", "L_main_C[5]", "L_main_C[6]",
                    "L_cross_C[17]", "L_cross_C[18]", "L_cross_C[19]", "L_cross_C[20]",
                    "L_cross_C[4]", "L_cross_C[5]", "L_cross_C[6]", "L_cross_C[10]", "L_cross_C[11]", "L_cross_C[12]", 
                    "L_main_C[7]", "L_main_C[8]", "L_main_C[9]", "L_main_C[10]",
                    "psi[1]", "psi[2]", "psi[3]", "psi[4]", "psi[5]", "psi[6]", "psi[7]", "psi[8]", "psi[9]", "psi[10]",
                    "phi_C[1,2]", "phi_C[1,3]", "phi_C[2,3]")

estdf.reg <- cbind.data.frame("par" = names(est.reg), 
                              "mean" = est.reg,
                              "2.5%" = NA,
                              "97.5%" = NA,
                              "prior" = "regsem")

## Add lavaan estimates
fit.lav0 <- cfa(HSmod0, data = traindat, std.lv = TRUE)
est.lav <- coef(fit.lav0)

names(est.lav) <- c("L_main_C[1]", "L_main_C[2]", "L_main_C[3]", "L_main_C[4]", "L_main_C[5]",
                    "L_main_C[6]", "L_main_C[7]", "L_main_C[8]", "L_main_C[9]", "L_main_C[10]",
                    "psi[1]", "psi[2]", "psi[3]", "psi[4]", "psi[5]", "psi[6]", "psi[7]", "psi[8]", "psi[9]", "psi[10]",
                    "phi_C[1,2]", "phi_C[1,3]", "phi_C[2,3]")
ci <- parameterEstimates(fit.lav0)[-c(21:23), ]
estdf.lav.main <- cbind.data.frame("par" = names(est.lav), 
                              "mean" = est.lav,
                              "2.5%" = ci[, "ci.lower"],
                              "97.5%" = ci[, "ci.upper"],
                              "prior" = "lavaan")
estdf.lav.cl <- cbind.data.frame("par" = paste0("L_cross_C[", 1:20, "]"),
                                 "mean" = 0,
                                 "2.5%" = NA,
                                 "97.5%" = NA,
                                 "prior" = "lavaan")
estdf.lav <- rbind.data.frame(estdf.lav.main, estdf.lav.cl)

## Combine with shrinkage estimates and plot
estdf.comb <- rbind.data.frame(estdf[which(estdf$prior != "reghs1"), ], # exclude reghs1 for now (no substantial differences with reghs2)
                               estdf.reg, 
                               estdf.lav)

# change factor levels
estdf.comb$prior <- as.factor(estdf.comb$prior)
levels(estdf.comb$prior) <- list("Classical" = "lavaan",
                                 "Classical lasso" = "regsem",
                                 "Bayesian ridge" = "ridge1",
                                 "Bayesian horseshoe" = "reghs2")

estdf.comb$par <- as.factor(estdf.comb$par)
levels(estdf.comb$par) <- list("V=~x1" = "L_main_C[1]","V=~x2" = "L_main_C[2]","V=~x3" = "L_main_C[3]",
                               "V=~x4" = "L_cross_C[7]","V=~x5" = "L_cross_C[8]","V=~x6" = "L_cross_C[9]",
                               "V=~x7" = "L_cross_C[13]","V=~x8" = "L_cross_C[14]","V=~x9" = "L_cross_C[15]","V=~x10" = "L_cross_C[16]",
                               "T=~x1" = "L_cross_C[1]","T=~x2" = "L_cross_C[2]","T=~x3" = "L_cross_C[3]",
                               "T=~x4" = "L_main_C[4]","T=~x5" = "L_main_C[5]","T=~x6" = "L_main_C[6]",
                               "T=~x7" = "L_cross_C[17]","T=~x8" = "L_cross_C[18]","T=~x9" = "L_cross_C[19]","T=~x10" = "L_cross_C[20]",
                               "S=~x1" = "L_cross_C[4]","S=~x2" = "L_cross_C[5]","S=~x3" = "L_cross_C[6]",
                               "S=~x4" = "L_cross_C[10]","S=~x5" = "L_cross_C[11]","S=~x6" = "L_cross_C[12]",
                               "S=~x7" = "L_main_C[7]","S=~x8" = "L_main_C[8]","S=~x9" = "L_main_C[9]","S=~x10" = "L_main_C[10]")
# TODO: add other parameters and change crossF below


# plot
pd <- position_dodge(0.3)

plotdat1 <- estdf.comb[which(estdf.comb$par %in% crossF1), ]
ggplot(plotdat1, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd) +
  theme_bw(base_size = 14, base_family = "") + 
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title = "", x = "Parameter", y = "Estimate")

plotdat2 <- estdf.comb[which(estdf.comb$par %in% crossF2), ]
ggplot(plotdat2, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd) +
  theme_bw(base_size = 14, base_family = "") + 
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title = "", x = "Parameter", y = "Estimate")

plotdat3 <- estdf.comb[which(estdf.comb$par %in% crossF3), ]
ggplot(plotdat3, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd) +
  theme_bw(base_size = 14, base_family = "") + 
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title = "", x = "Parameter", y = "Estimate")

plotdat4 <- estdf.comb[grep("L_main_C", estdf.comb$par), ]
ggplot(plotdat4, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd) +
  theme_bw(base_size = 14, base_family = "") + 
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title = "", x = "Parameter", y = "Estimate")

plotdat5 <- estdf.comb[grep("phi_C", estdf.comb$par), ]
ggplot(plotdat5, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd) +
  theme_bw(base_size = 14, base_family = "") + 
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title = "", x = "Parameter", y = "Estimate")

plotdat6 <- estdf.comb[grep("psi", estdf.comb$par), ]
ggplot(plotdat6, aes(x = par, y = `mean`, colour = prior, group = prior)) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = .2, position = pd) +
  geom_point(position = pd) +
  theme_bw(base_size = 14, base_family = "") + 
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title = "", x = "Parameter", y = "Estimate")

