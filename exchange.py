import math
import numpy
from scipy.stats import norm


class Exchange:
  def __init__(self,prob,price1,price2,maturity):
    self.prob = prob
    self.price1 = price1
    self.price2 = price2
    self.maturity = maturity
  
  def payoff(self):
    return max(0,(self.price2.value-self.price1.value))
    
  def approximate(self,t):#actually its exact in this case
    sigma1 = self.price1.volatility
    sigma2 = self.price2.volatility
    rho12 = self.prob.getCorrelation(self.price1,self.price2)
    
    sigmat = math.sqrt(sigma1**2+sigma2**2-2*sigma1*sigma2*rho12)
    
    r=self.price1.drift
    tau=self.maturity-t
    
    dplus=(math.log(self.price2.approximate(t)/self.price1.approximate(t))+tau*(sigmat**2)/2)/(sigmat*math.sqrt(tau))
    dmin=(math.log(self.price2.approximate(t)/self.price1.approximate(t))-tau*(sigmat**2)/2)/(sigmat*math.sqrt(tau))
    
    return self.price2.approximate(t)*norm.cdf(dplus)-self.price1.approximate(t)*norm.cdf(dmin)  