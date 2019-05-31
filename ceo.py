import math
import numpy
from scipy.stats import norm

class CEO:
  def __init__(self,prob,call1,call2):
    self.call1 = call1
    self.call2 = call2
    self.maturity = call1.maturity
    self.prob = prob
    self.nu = 1
    
  def payoff(self):
    return max(0,self.call1.payoff()-self.call2.payoff())
    # return max(0,self.call1.blackScholes(T)-self.call2.blackScholes(T))
  
  def approximate3(self,t):
    r = self.call1.price.drift
    sigma1 = self.call1.price.volatility
    sigma2 = self.call2.price.volatility
    
    delta1 = self.call1.partialDerivative(0,self.call1.price)
    delta2 = self.call2.partialDerivative(0,self.call2.price)
  
    k1 = (self.call1.approximate(0)/(self.call1.price.approximate(t) * delta1))-1
    k2 = (self.call2.approximate(0)/(self.call2.price.approximate(t) * delta2))-1
    
    rho = self.prob.getCorrelation(self.call1.price,self.call2.price)
    correlations = numpy.matrix([[1,rho],[rho,1]],dtype='float64')
    
    volatilities = [sigma1/(1+k1),sigma2/(1+k2)]
    
    drifts = [0.04]*2
    initials = [self.call1.approximate(0),self.call2.approximate(0)]
    newprob=Probability(correlations,initials,drifts,volatilities)
    
    xchg=Exchange(newprob,newprob.prices[1],newprob.prices[0],self.maturity)
    
    return xchg.approximate(t)
    
  def approximate2(self,t):
    amount=100000
    total=0
    T=self.maturity
    
    for i in range(amount):  
      self.prob.samplePrices(0)    
      r = self.call1.price.drift
      sigma1 = self.call1.price.volatility
      sigma2 = self.call2.price.volatility
      
      delta1 = self.call1.partialDerivative(0,self.call1.price)
      delta2 = self.call2.partialDerivative(0,self.call2.price)
      
      gamma1 = self.call1.gamma(0)
      gamma2 = self.call2.gamma(0)
      
      c1 = (sigma1 * self.call1.price.approximate(t) * gamma1)/delta1
      c2 = (sigma2 * self.call2.price.approximate(t) * gamma2)/delta2
      
      sigmat1 = sigma1 + c1
      sigmat2 = sigma2 + c2

      W1Ster = self.prob.motions[0].getSampled()-0.5*sigmat1*t
      W2Ster = self.prob.motions[1].getSampled()-0.5*sigmat2*t
    
      k1 = (self.call1.approximate(0)/(self.call1.price.approximate(t) * delta1))-1
      k2 = (self.call2.approximate(0)/(self.call2.price.approximate(t) * delta2))-1
      
      U1 = self.call1.approximate(0) * math.exp(r*T) * (math.exp(sigmat1 *W1Ster)+k1)/(1+k1)
      U2 = self.call2.approximate(0) * math.exp(r*T) * (math.exp(sigmat2 *W2Ster)+k2)/(1+k2)
    
      total+=math.exp(-r*T)*max(U1-U2,0)
    
    return total/amount
  
  def approximate(self,t):   #for now we assume t is 0
    C1=self.call1.approximate(t) 
    C2=self.call2.approximate(t)
    
    P1=0
    P2=0
    
    r = self.call1.price.drift
    tau = self.maturity-t
    
    S1 = self.call1.price.approximate(t)
    S2 = self.call2.price.approximate(t)
    sigma1 = self.call1.price.volatility
    sigma2 = self.call2.price.volatility
    rho12 = self.prob.getCorrelation(self.call1.price,self.call2.price)
    
    pdv1 = self.call1.partialDerivative(t,self.call1.price)
    pdv2 = self.call2.partialDerivative(t,self.call2.price)
    
    sigmat1 = 0 if C1==0 else  self.nu * math.sqrt((pdv1**2) * (sigma1 **2) * (S1**2))/C1
    sigmat2 = 0 if C2==0 else  self.nu * math.sqrt((pdv2**2) * (sigma2 **2) * (S2**2))/C2
    
    if (sigmat1==0 or sigmat2==0):
      gamma1=sigmat1+sigmat2
    else:
      beta12 = (rho12*sigma1*sigma2*S1*S2*pdv1*pdv2)/(sigmat1*sigmat2*C1*C2)
      gamma1 = math.sqrt(sigmat1**2+sigmat2**2-2*sigmat1*sigmat2*beta12)
    
    if (C1!=0 and C2!=0):
      d1plus = (math.log(C1/C2) + tau*(gamma1**2)/2)/(gamma1*math.sqrt(tau))
      d1min = (math.log(C1/C2) - tau*(gamma1**2)/2)/(gamma1*math.sqrt(tau))
    else:
      d1plus=math.inf
      d1min=math.inf
    
    
    E1 = C1*norm.cdf(d1plus) - C2* norm.cdf(d1min)
    E2 = 0
    
    return E1+E2