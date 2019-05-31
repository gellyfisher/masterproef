import math
import numpy
from scipy.stats import norm
import scipy.integrate as integrate
import time

from prices import *

class Option: #a plain call (theta=1) or put option (theta=-1)
  def __init__(self,strike,price,maturity,theta=1):
    self.strike = strike
    self.price = price
    self.theta = theta
    self.maturity = maturity
    
    self.partialCache = {}
    self.approxCache = {}
  
  def payoff(self,approx=False):
    if not approx:
      return max(0,self.theta*(self.price.value-self.strike))
    else:
      return max(0,self.theta*(self.price.approximate(self.maturity)-self.strike))
    
  def approximate(self,t):#precies indien t=0 dit geval 
    if (t in self.approxCache):
      return self.approxCache[t]
    elif t==self.maturity:
      return self.payoff(True)
    else:
      s = 1 if self.theta>=0 else -1
      r=self.price.drift
      
      tau=self.maturity-t
      
      if (callable(self.price.volatility)): #we hebben een niet constante volatiliteit
        sigma = math.sqrt(integrate.quad(lambda t:self.price.volatility(t)**2,0,tau)[0]/tau)
      else:
        sigma = self.price.volatility
      
      if (self.strike>0):
        dplus=(math.log(self.price.approximate(t)/self.strike)+(r+(sigma**2)/2)*tau)/(sigma*math.sqrt(tau))
        dmin=(math.log(self.price.approximate(t)/self.strike)+(r-(sigma**2)/2)*tau)/(sigma*math.sqrt(tau))
      else:
        dplus=math.inf
        dmin=math.inf
        
      self.approxCache[t] = self.theta*(self.price.approximate(t)*norm.cdf(s*dplus)-math.exp(-tau*r)*self.strike*norm.cdf(s*dmin))
      return self.approxCache[t]
      
  def partialDerivative(self,t,price):
    if (price!=self.price):
      return 0
    else:
      if (t in self.partialCache):
        return self.partialCache[t]
      else:
        s = 1 if self.theta>=0 else -1
        r=self.price.drift
        sigma=self.price.volatility
        tau=self.maturity-t
        
        if (self.strike>0 and tau>0):
          dplus=(math.log(self.price.approximate(t)/self.strike)+(r+(sigma**2)/2)*tau)/(sigma*math.sqrt(tau))
        elif tau==0 and self.price.approximate(t)<self.strike:
          dplus=-math.inf
        else:
          dplus=math.inf
        
        self.partialCache[t] = self.theta * norm.cdf(s*dplus)     
        return self.partialCache[t]
    
  def gamma(self,t):
    s = 1 if self.theta>=0 else -1
    r=self.price.drift
    sigma=self.price.volatility
    tau=self.maturity-t
    
    dplus=s*(math.log(self.price.approximate(t)/self.strike)+(r+(sigma**2)/2)*tau)/(sigma*math.sqrt(tau))
    
    return self.theta * (math.exp(-0.5*dplus**2))/(math.sqrt(2*math.pi*tau)*sigma*self.price.approximate(t))
