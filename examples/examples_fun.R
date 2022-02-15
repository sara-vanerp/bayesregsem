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
