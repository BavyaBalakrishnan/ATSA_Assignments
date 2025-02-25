install.packages(c("psych","tseries","matrixTests","fDMA","urca","rmgarch","xts","MTS","riskR"))
library(psych)
library(matrixTests)
library(tseries)
library(fDMA)
library(urca)
#rmgarch for multivariate models
library(rmgarch)
library(xts)
library(MTS)
library(riskR)
library(readxl)
#Read excel containing daywise N50 returns and Oil returns
Oil_stock <- read_excel("Oil_stock.xls")
#DCC-AR (1)-multivariate GARCH(1,1) with or without asymmetry
DCC_AR_GARCH_WITH_OR_WITHOUT_ASYMMETRY <- function(rX1,varmodel)
{
  #specifying univariate model we want to estimate
  ug_spec <- ugarchspec(mean.model=list(armaOrder=c(1,0)))
  print(ug_spec)
  #Fitting with univariate garch model
  ugfit = ugarchfit(spec = ug_spec, solver='hybrid',data = Oil_stock$Daily_returns_Stocks)
  #print(ugfit)
  #multivariate garch model.We are using the same univariate volatility model for each of the asset
  #By using replicate(2, ugarchspec...) we replicate the model 2 times
  uspec.n = multispec(replicate(2, ugarchspec(mean.model = list(armaOrder = c(1,0)),variance.model = list(model = varmodel))))
  #fit the data to multivariate garch model
  multf = multifit(uspec.n,rX1)
  #you can type multf into the command window to see the estimated parameters for the model
  #to specify the correlation specification we use the dccspec function
  #In this specification we can to specify how the univariate volatilities are modeled (as per uspec.n) and how complex the dynamic structure of the correlation matrix is (here we are using the most standard dccOrder = c(1, 1) specification).
  spec1 = dccspec(uspec = uspec.n, dccOrder = c(1, 1), distribution = 'mvnorm')
  #estimate the model using the dccfit function.
  #rX:data , fit.control = list(eval.se = TRUE) : ensures  the estimation procedure produces standard errors for estimated parameters., multf: already estimated multivariate model
  fit1 = dccfit(spec1,data = rX1,fit.control = list(eval.se = TRUE), fit = multf)
  print(fit1)
  #find the dynamic correlation and covariance
  cov1 = rcov(fit1) # extracts the covariance matrix
  cor1 = rcor(fit1) # extracts the correlation matrix. it contains the correlation between assets for each day
  #plot the correlation between OIL and N50
  par(mfrow=c(1,1))
  
  cor_OS <- cor1[1,2,]
  cor_OS <- as.xts(cor_OS)
  cor_OS <- ts(cor_OS,start=c(2006,191),frequency=365)
  plot(cor_OS)
  
  
}
#CCC-AR (1)-multivariate GARCH(1,1) with or without asymmetry
CCC_AR_GARCH_WITH_OR_WITHOUT_ASYMMETRY <- function(rX1,varmodel)
{
  uspec = ugarchspec(mean.model = list(armaOrder = c(1,0)), variance.model = list(model = varmodel))
  #specify the CCC MGARCH model.  cgarchspec :Method for creating a Copula-GARCH specification object prior to fitting.
  spec = cgarchspec(uspec = multispec( replicate(2, uspec) ), VAR = FALSE,  
                    dccOrder = c(1,1), distribution.model = list(copula = c("mvnorm"), method = c("ML"),
                                                                 time.varying = FALSE, transformation = "parametric"), start.pars = list(), fixed.pars = list())
  fit1 = cgarchfit(spec, data = rX1,  fit.control = list(eval.se = TRUE, trace = TRUE), solver = "solnp")
  
  print(fit1)
  #find the constant covariance and correlation
  cov1 = rcov(fit1)
  cor1 = rcor(fit1)
  #print the constant correlation between indices
  print(cor1)
}
#DCC-VARMA (1,1)-multivariate GARCH(1,1) with or without asymmetry
DCC_VARMA_GARCH_WITH_OR_WITHOUT_ASYMMETRY<- function(rX1,varmodel)
{
  #Performs conditional maximum likelihood estimation of a VARMA model
  #p -AR order, q -MA order. Include the mean in estimation.
  m2=VARMA(rX1,p=1,q=1,include.mean=TRUE)
  print(m2) #residuals:Residual matrix,Sigma:Residual covariance matrix, Phi:VAR coefficients, Theta:VMA coefficients,Ph0:The constant vector
  #univariate VARMA-GARCH
  ug_spec <- ugarchspec(mean.model=list(m2))
  #print(ug_spec)
  #multivariate VARMA-GARCH
  uspec.n = multispec(replicate(2, ugarchspec(variance.model=list(model=varmodel,
                                                                  garchOrder=c(1,1)),mean.model = list(m2,include.mean=TRUE))))
  multf = multifit(uspec.n, rX1)
  spec1 = dccspec(uspec = uspec.n, dccOrder = c(1, 1), distribution = 'mvnorm')
  fit1 = dccfit(spec=spec1, data = rX1,out.sample =0, solver = "solnp",fit.control = list(eval.se = TRUE), fit = multf)
  print(fit1)
  #find the dynamic covariance and correlation
  cov1 = rcov(fit1)
  cor1 = rcor(fit1)
  #plot the correlation between OIL and N50
  par(mfrow=c(1,1))
  
  cor_OS <- cor1[1,2,]
  cor_OS <- as.xts(cor_OS)
  cor_OS <- ts(cor_OS,start=c(2006,191),frequency=365)
  plot(cor_OS)
  return (fit1)
}

#CCC-VARMA (1,1)-multivariate GARCH(1,1) with or without asymmetry
CCC_VARMA_GARCH_WITH_WITHOUT_ASYMMETRY<- function(rX1,varmodel)
{
  m2=VARMA(rX1,p=1,q=1,include.mean=TRUE)
  #univariate VARMA-GARCH
  uspec.n = multispec(replicate(2, ugarchspec(variance.model=list(model=varmodel,
                                                                  garchOrder=c(1,1)),mean.model = list(m2,include.mean=TRUE))))
  #multivariate CCC-VARMA-MGARCH
  spec = cgarchspec(uspec = uspec.n, VAR = TRUE,  
                    dccOrder = c(1,1), distribution.model = list(copula = c("mvnorm"), method = c("ML"),
                                                                 time.varying = FALSE, transformation = "parametric"), start.pars = list(), fixed.pars = list())
  fit1 = cgarchfit(spec, data = rX1, spd.control = list(lower = 0.1, upper = 0.9, type = "pwm",
                                                        kernel = "epanech"), fit.control = list(eval.se = TRUE, trace = TRUE), solver = "solnp")
  print(fit1)
  cov1 = rcov(fit1)
  cor1 = rcor(fit1)
  #plot the correlation between OIL and N50
  print(cor1)
  
}


DESCRIPTIVE_STATISTICS<-function()
{  
  #descriptive statistics of CRUDE OIL and N50
  print("OIL")
  print(describe(Oil_stock$Daily_returns_oil))
  print("N50")
  print(describe(Oil_stock$Daily_returns_Stocks))
  
  #Jarque-Bera Test for normality
  #test1
  print("Jarque-Bera Test for normality(Test1)")
  print("OIL")
  print(col_jarquebera(Oil_stock$Daily_returns_oil))
  print("N50")
  print(col_jarquebera(Oil_stock$Daily_returns_Stocks))
  #Lagrange Multiplier (LM) test for autoregressive conditional heteroscedasticity (ARCH)  Engle's LM ARCH Test
  print("Lagrange Multiplier (LM) test for ARCH")
  print("OIL")
  wti <-Oil_stock$Daily_returns_oil
  arch1 <- archtest(ts=as.vector(wti),lag=10)
  print(arch1)
  print("N50")
  wti <-Oil_stock$Daily_returns_Stocks
  arch3 <- archtest(ts=as.vector(wti),lag=10)
  print(arch3)
  
  #unit root tests to check stationarity of timeseries
  #ADF test
  
  print("OIL")
  y_none = ur.df(Oil_stock$Daily_returns_oil,type = "none", selectlags = "AIC")
  print(summary(y_none))
  print("N50")
  y_none = ur.df(Oil_stock$Daily_returns_Stocks,type = "none", selectlags = "AIC")
  print(summary(y_none))
  
  #PP Test
  print("OIL")
  dp_constant = ur.pp(Oil_stock$Daily_returns_oil,type = "Z-tau", model="constant", lags = "short")
  print(summary(dp_constant))
  print("N50")
  dp_constant = ur.pp(Oil_stock$Daily_returns_Stocks,type = "Z-tau", model="constant", lags = "short")
  print(summary(dp_constant))
  
  #KPSS test
  
  print("OIL")
  k_test = ur.kpss(Oil_stock$Daily_returns_oil,type = "tau", lags = "short")
  print(summary(k_test))
  print("N50")
  k_test = ur.kpss(Oil_stock$Daily_returns_Stocks,type = "tau", lags = "short")
  print(summary(k_test))
  
  #Q for finding serial correlations
  print("OIL")
  data=Oil_stock$Daily_returns_oil
  fit1=arima(data,order=c(1,0,0))
  print(Box.test(resid(fit1),type="Ljung",lag=20,fitdf=1))
  print("N50")
  data=Oil_stock$Daily_returns_Stocks
  fit1=arima(data,order=c(1,0,0))
  print(Box.test(resid(fit1),type="Ljung",lag=20,fitdf=1))
}

#Cointegration Tests
CONINTEGRATION_TESTS<-function()
{
  #Engle and Granger (1987) cointegration tests :put type="trend" for the test with trend
  #creating residuals based on the static regression and then testing the residuals for the presence of unit roots.
  print("OIL~N50")
  longrun=lm(Oil_stock$Daily_returns_Stocks~Oil_stock$Daily_returns_oil)
  resid_longrun=longrun$residuals
  #ADF tests to test for stationarity units in time series.
  y=ur.df(resid_longrun,type="none",selectlags = "AIC")
  print(summary(y))
  
  #Johansen cointegration tests
  
  jotest=ca.jo(data.frame(Oil_stock$Daily_returns_Stocks,Oil_stock$Daily_returns_oil), type="trace", K=2, ecdet="none", spec="longrun")
  print(summary(jotest))
  
  #Phillips-Ouliaris cointegration test
  
  dat<-data.frame(Oil_stock$Daily_returns_Stocks,Oil_stock$Daily_returns_oil)
  
  print(po.test(dat, demean = TRUE, lshort = FALSE))
  print(ca.po(dat, demean = "constant", lag = "long", type = "Pu"))
  print(ca.po(dat, demean = "constant", lag = "long", type = "Pz"))
  
}


#forecast correlation between OIL and N50
FORECAST_CORRELATION<-function(bestmodel)
{
  cor1 = rcor(bestmodel)
  cov1 = rcov(bestmodel)
  
  #forecasts for the covariance or correlation matrix
  dccf1 <- dccforecast(bestmodel, n.ahead = 20)
  #DCC volatility(sigma) unconditional forecast
  #plot(dccf1, which = 1, series=c(1,2)) 
  plot(dccf1, which = 2, series=c(1,2))
  #actual forecasts for the correlation
  Rf <- dccf1@mforecast$R #H for the covariance forecast
  # printing the structure of Rf. object Rf is a list with one element. one list item is then a 2 dimensional matrix/array which contains the the 30 forecasts of 2×2 correlation matrices.
  str(Rf)
  corf_OS <- Rf[[1]][1,2,] #Correlation forecasts between Oil and N50. [ [1] ] tells R to go to the first list item and then [1,2,] instructs R to select the (1,2) element of all available correlation matrices.
  print("Forecast of correlation between Oil and N50 for 20 days")
  #Forecast of correlation between Oil and N50 for 20 days
  print(corf_OS)
  #plot the forcasted correlation between OIL and N50. display the forecast along with the last in-sample estimates of correlation.
  c_OS <- c(tail(cor1[1,2,],30),rep(NA,20)) #gets the last 30 correlation observations
  cf_OS <- c(rep(NA,30),corf_OS) # gets the 30 forecasts
  plot(c_OS,type = "l",main="Correlation between CRUDE OIL and N50")
  lines(cf_OS,type = "l", col = "orange")
  
  
}





#plot the timeseries Crude OIL and N50
par(mfrow=c(2,1))
OIL<-ts(Oil_stock$Daily_returns_oil,start=c(2006,191),frequency=365)
plot(OIL)
N50<-ts(Oil_stock$Daily_returns_Stocks,start=c(2006,191),frequency=365)
plot(N50)

#descriptive statistics of daily returns of Crude OIL and N50
DESCRIPTIVE_STATISTICS()

#Cointegration tests for Crude OIL and N50
CONINTEGRATION_TESTS()

rX1 <- data.frame(Oil_stock$Daily_returns_oil,Oil_stock$Daily_returns_Stocks)
names(rX1)[1] <- "Crude_OIL"
names(rX1)[2] <- "N50"

#AR (1)-multivariate GARCH(1,1) DCC WITHOUT ASYMMETRY
#sGARCH : standard GARCH model
#alpha: ARCH parameter, beta: GARCH parameter, variance intercept parameter is ’omega
DCC_AR_GARCH_WITH_OR_WITHOUT_ASYMMETRY(rX1,varmodel="sGARCH")

#AR (1)-multivariate GARCH(1,1) DCC WITH ASYMMETRY
#The GJR GARCH model of Glosten et al. (1993) models positive and negative shocks on the
#conditional variance asymmetrically via the use of the indicator function I
DCC_AR_GARCH_WITH_OR_WITHOUT_ASYMMETRY(rX1,varmodel="gjrGARCH")

#AR (1)-multivariate GARCH(1,1) CCC WITHOUT ASYMMETRY
CCC_AR_GARCH_WITH_OR_WITHOUT_ASYMMETRY(rX1,varmodel="sGARCH")

#AR (1)-multivariate GARCH(1,1) CCC WITH ASYMMETRY
CCC_AR_GARCH_WITH_OR_WITHOUT_ASYMMETRY(rX1,varmodel="gjrGARCH")

#VARMA (1,1)–multivariate GARCH(1,1) DCC WITHOUT ASYMMETRY
Model<-DCC_VARMA_GARCH_WITH_OR_WITHOUT_ASYMMETRY(rX1,varmodel="sGARCH")

#VARMA (1,1)–multivariate GARCH(1,1) DCC WITH ASYMMETRY
bestmodel<-DCC_VARMA_GARCH_WITH_OR_WITHOUT_ASYMMETRY(rX1,varmodel="gjrGARCH")

#VARMA (1,1)–multivariate GARCH(1,1) CCC WITHOUT ASYMMETRY
CCC_VARMA_GARCH_WITH_WITHOUT_ASYMMETRY(rX1,varmodel="sGARCH")

#VARMA (1,1)–multivariate GARCH(1,1) CCC WITH ASYMMETRY
CCC_VARMA_GARCH_WITH_WITHOUT_ASYMMETRY(rX1,varmodel="gjrGARCH")

#plot volatility for OIL,N50
vol<-sigma(bestmodel)
volatility <- ts(vol,start=c(2006,191),frequency=365)
plot(volatility)
cor1 = rcor(bestmodel)
cov1 = rcov(bestmodel)

#forecast correlation between and Crude oil and N50
# use the best estimated model to produce forecasts for the covariance or correlation matrix
FORECAST_CORRELATION(bestmodel)
#Various Risk measures
print("Risk measures of N50/OIL")
risk.hedge(Oil_stock$Daily_returns_Stocks, Oil_stock$Daily_returns_oil, c(0.01, 0.05))

