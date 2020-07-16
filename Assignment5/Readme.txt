How COVID 19 pandemic impact on stock prices and find out correlation between COVID 19 and stock price and develop machine learning or statistical model for predict the same.
Covid19 data collected from https://covid19.who.int/ provided by WHO. The number of confirmed cases over time is chosen for finding correlation. The proxy for the Indian stock market is the S&P
CNX Nifty 50 index (N50) obtained from National Stock Exchange of India. Used the daily data for both variables from 31-01-2006 to 22-05-2020. Computed the continuously compounded daily
returns by calclulating the difference in the logarithmic values of two consecutive prices: 
r i ,t = ln( Pi ,t / P i ,t-1 ) × 100 , where r i , t indicates the continuously compounded percentage daily returns for index i at time t , while P i ,t indicates the price level of index i at time t .
Language used: R
Libraries :rmgarch,xts,psych,matrixTests,tseries,fDMA,urca,riskR,readxl,MTS
In just weeks, the Corona virus pandemic has shaved off nearly a third of the global market cap. The spread of the virus has triggered panic across the world and shaken the confidence of investors.
On 28th February the Indian indices registered a 3.5% fall which was the second-biggest fall in the history of the Sensex. The stock market has historically been prone to fear psychoses, and this is
one such instance. The automobile and healthcare industry are significant stakeholders in the Indian stock market. If their operations and production get affected due to the Coronavirus outbreak and
China’s lockdown, it could lead to reduced investor faith in the market.
