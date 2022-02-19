##### BAYESREGSEM EXAMPLES - FUNCTIONS #####
## Author: Sara van Erp

## Function to generate data based on covariance matrix
simdat.fun <- function(nobs, nitem, covmat, varnames){
  simdat <- MASS::mvrnorm(n = nobs, mu = rep(0, nitem), Sigma = covmat)
  colnames(simdat) <- varnames
  out <- simdat
  return(out)
}

## Function to fit the models and return fit and modification indices
fit.fun <- function(mod, data){
  fit <- cfa(mod, data = data)
  fitind <- fitmeasures(fit)[c("pvalue", "cfi", "tli", "rmsea", "srmr")]
  modind <- modindices(fit, sort = TRUE)
  out <- list("fit" = fitind,
              "MIs" = modind)
  return(out)
}

## Function to check if the model fits
# Use the following cutoffs for good fit:
# CFI > .95 & TLI > .90 & RMSEA < 05 & SRMR < .08 and chisquare p-value > .05 if chisq = TRUE
checkfit.fun <- function(fitind, cfi = .95, tli = .90, rmsea = .05, srmr = .08, chisq = TRUE){
  if(chisq == TRUE){
    if(fitind["pvalue"] > .05 & fitind["cfi"] > cfi & fitind["tli"] > tli & fitind["rmsea"] < rmsea & fitind["srmr"] < srmr){
      out <- "good fit"
    } else{
      out <- "bad fit"
    }
  } 
  if(chisq == FALSE){
    if(fitind["cfi"] > cfi & fitind["tli"] > tli & fitind["rmsea"] < rmsea & fitind["srmr"] < srmr){
      out <- "good fit"
    } else{
      out <- "bad fit"
    }
  }
  
  return(out)
}

## Function to automatically adapt the model based on the highest MI
# Only cross-loadings are selected for now to have a clear comparison with the shrinkage priors
adapt.fun <- function(mi, mod){
  
  # select only cross-loadings
  misel <- mi[mi$op == "=~", ]
  
  # add the first cross-loading to free to the model
  addline <- paste(misel[1, "lhs"], misel[1, "op"], misel[1, "rhs"])
  modnew <- paste(mod, addline, sep = " \n ")
  return(modnew)
}

## Function to create a dataframe with posterior means and CIs based on list with stan fitobjects
df_est <- function(fitls, CI = "95%"){
  out <- lapply(fitls, function(x){
    summ <- summary(x, probs = c(0.025, 0.975, 0.05, 0.95, 0.1, 0.9))$summary
    if(CI == "95%"){
      cbind.data.frame("par" = rownames(summ), 
                       summ[, c("mean", "2.5%", "97.5%")])
    } else if(CI == "90%"){
      cbind.data.frame("par" = rownames(summ), 
                       summ[, c("mean", "5%", "95%")])
    } else if(CI == "80%"){
      cbind.data.frame("par" = rownames(summ), 
                       summ[, c("mean", "10%", "90%")])
    }
    
  })
  out.df <- list()
  for(i in 1:length(out)){
    prior <- names(out)[i]
    out.df[[i]] <- cbind.data.frame(out[[i]], prior)
  }
  return(out.df)
}
