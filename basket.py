import math
import numpy
from scipy.stats import norm
from scipy.stats import gamma
import scipy.integrate as integrate

from prices import *
from option import *

class Basket:
  def __init__(self,prob,strike,prices,maturity,thetas,nu=1,method="calibrate",xAmount=100,productmethod=False):
    self.prob = prob          
    self.prices = prices
    self.thetas = thetas
    self.maturity = maturity
    self.nu=nu
    
    self.method = method
    self.xAmount = xAmount
    self.productmethod = productmethod
    
    # we hebben een cache om te voorkomen dat we heel vaak dezelfde berekeningen opnieuw doen. 
    # we houden als key de tijd bij en als value de relevante waarde
    self.dplusminCache = {}  
    self.partialCache = {}
    self.approxCache = {}
    self.volatilitiesCache = {}
    self.correlationsCache = {}
    
    self.setStrike(strike)
  
  #nadat de strike of nu verandert dan veranderen alle resultaten dus we mogen de cache zeker niet hergebruiken
  def resetCache(self):
    self.dplusminCache = {}
    self.partialCache = {}
    self.approxCache = {}
    self.volatilitiesCache = {}
    self.correlationsCache = {}
    
  def setNu(self,nu):
    self.nu=nu
    self.resetCache()
    self.split()
    
  def setStrike(self,strike):
    if isinstance(strike, list):
      self.strikes = strike
    else:
      m=numpy.count_nonzero(self.thetas)
      self.strikes = []
      for i in range(len(self.thetas)):
        if self.thetas[i]==0:
          self.strikes.append(0)
        else:
          self.strikes.append(strike/(self.thetas[i]*m))

    self.resetCache()
    self.split()
      
  def payoff(self,approx=False):
    if not approx:
      temp = sum([self.thetas[i]*(self.prices[i].value-self.strikes[i]) for i in range(len(self.prices))])
    else:
      temp = sum([self.thetas[i]*(self.prices[i].approximate(self.maturity)-self.strikes[i]) for i in range(len(self.prices))])
    return max(0,temp)
    
  def split(self):
    N=len(self.thetas)
    
    thetas1 = self.thetas[:N//2]
    strikes1 = self.strikes[:N//2] 
    prices1 = self.prices[:N//2]
    
    thetas2=self.thetas[N//2:]
    strikes2 = self.strikes[N//2:]
    prices2 = self.prices[N//2:]
  
    if (N>=4):
      self.C1=Basket(self.prob,strikes1,prices1,self.maturity,thetas1,self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      self.C2=Basket(self.prob,strikes2,prices2,self.maturity,[-1*t for t in thetas2],self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      
      self.P1=Basket(self.prob,strikes1,prices1,self.maturity,[-1*t for t in thetas1],self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      self.P2=Basket(self.prob,strikes2,prices2,self.maturity,thetas2,self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      
    elif (N==3):
      self.C1=Option(strikes1[0],prices1[0],self.maturity,thetas1[0])
      self.C2=Basket(self.prob,strikes2,prices2,self.maturity,[-1*t for t in thetas2],self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      
      self.P1=Option(strikes1[0],prices1[0],self.maturity,-thetas1[0])
      self.P2=Basket(self.prob,strikes2,prices2,self.maturity,thetas2,self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      
    elif (N==2):
      self.C1=Option(strikes1[0],prices1[0],self.maturity,thetas1[0])
      self.C2=Option(strikes2[0],prices2[0],self.maturity,-thetas2[0])
      
      self.P1=Option(strikes1[0],prices1[0],self.maturity,-thetas1[0])
      self.P2=Option(strikes2[0],prices2[0],self.maturity,thetas2[0])
    #N==1 is een gewone optie.
    return N
    
  def getVolatilities(self,t,numeric=True):      
    if ((t,numeric) in self.volatilitiesCache):
      return self.volatilitiesCache[(t,numeric)]
    elif self.method=="calibrate" or not numeric:
      N = len(self.thetas)
      r = self.prices[0].drift
      tau = self.maturity-t
      
      #lemma 2.3.2
      sigmat = [0, 0]
      psit = [0, 0]
      
      m = [0,N//2,N]
      C=[self.C1,self.C2]
      P=[self.P1,self.P2]
      
      for i in range(2):
        for j in range(m[i],m[i+1]):
          for k in range(m[i],m[i+1]):
            sigmaj = self.prices[j].volatility
            sigmak = self.prices[k].volatility
            
            rhokj = self.prob.getCorrelation(self.prices[j],self.prices[k])
            
            pdvCj = C[i].partialDerivative(t,self.prices[j])
            pdvCk = C[i].partialDerivative(t,self.prices[k])
            
            pdvPj = P[i].partialDerivative(t,self.prices[j])
            pdvPk = P[i].partialDerivative(t,self.prices[k])
            
            if not self.productmethod:
              #hier gebruiken we E[Sk] * E[Sj]
              Sj = self.prices[j].approximate(t)
              Sk = self.prices[k].approximate(t)
              
              sigmat[i] += pdvCj * pdvCk * rhokj * Sk * Sj * sigmaj * sigmak
              psit[i] += pdvPj * pdvPk * rhokj * Sk * Sj * sigmaj * sigmak
              
            else:
              #we bepalen E[Sk * Sj]
              prod = self.prob.prices[0].initial*self.prob.prices[1].initial * math.exp((2*r + sigmaj*sigmak*rhokj)*t)        
              
              sigmat[i] += pdvCj * pdvCk * rhokj * prod * sigmaj * sigmak
              psit[i] += pdvPj * pdvPk * rhokj * prod * sigmaj * sigmak
      
      
      C1t=self.C1.approximate(t)
      C2t=self.C2.approximate(t)
      P1t=self.P1.approximate(t)
      P2t=self.P2.approximate(t)
    
      sigmat1 = 0 if C1t==0 else self.nu * math.sqrt(sigmat[0]) / C1t
      sigmat2 = 0 if C2t==0 else self.nu * math.sqrt(sigmat[1]) / C2t
      
      psit1 = 0 if P1t==0 else self.nu * math.sqrt(psit[0]) / P1t
      psit2 = 0 if P2t==0 else self.nu * math.sqrt(psit[1]) / P2t
              
    elif (t==self.maturity):
      sigmat1,sigmat2,psit1,psit2=self.getVolatilities(t,False)
    else:
      istart = round(t*self.xAmount/self.maturity)
      xArray=[(i*self.maturity)/self.xAmount for i in range(istart,self.xAmount)]
      #het kan door een numerieke fout gebeuren dat (i*self.maturity)/i niet gelijk is aan self.maturity wat voor problemen zorgt. daarom voegen we dit appart toe
      xArray.append(self.maturity)
      
      # print(xArray)
      
      sigmat1=math.sqrt(integrate.simps([self.getVolatilities(x,False)[0]**2 for x in xArray],xArray)/(self.maturity-t))
      sigmat2=math.sqrt(integrate.simps([self.getVolatilities(x,False)[1]**2 for x in xArray],xArray)/(self.maturity-t))
      
      psit1=math.sqrt(integrate.simps([self.getVolatilities(x,False)[2]**2 for x in xArray],xArray)/(self.maturity-t))
      psit2=math.sqrt(integrate.simps([self.getVolatilities(x,False)[3]**2 for x in xArray],xArray)/(self.maturity-t))
  
    self.volatilitiesCache[(t,numeric)]=(sigmat1,sigmat2,psit1,psit2)
    return (sigmat1,sigmat2,psit1,psit2)
      
  def getCorrelations(self,t,numeric=True):
    if ((t,numeric) in self.correlationsCache):
      return self.correlationsCache[(t,numeric)]
    elif self.method=="calibrate" or not numeric:
      sigmat1,sigmat2,psit1,psit2 = self.getVolatilities(t) 
      N=len(self.thetas)
      r=self.prices[0].drift
      
      beta12 = 0
      gamma12 = 0
      for j in range(0,N//2):
          for k in range(N//2,N):
            
            sigmaj = self.prices[j].volatility
            sigmak = self.prices[k].volatility
            rhokj = self.prob.getCorrelation(self.prices[j],self.prices[k])

            pdvCj = self.C1.partialDerivative(t,self.prices[j])
            pdvCk = self.C2.partialDerivative(t,self.prices[k])
            
            pdvPj = self.P1.partialDerivative(t,self.prices[j])
            pdvPk = self.P2.partialDerivative(t,self.prices[k])
            
            if not self.productmethod:
              #hier gebruiken we E[Sk] * E[Sj]
              Sj = self.prices[j].approximate(t)
              Sk = self.prices[k].approximate(t)
              
              beta12 += pdvCj * pdvCk * rhokj * Sk * Sj * sigmaj * sigmak
              gamma12 += pdvPj * pdvPk * rhokj * Sk * Sj * sigmaj * sigmak
            else:
              #we bepalen E[Sk * Sj]
              prod = self.prob.prices[0].initial*self.prob.prices[1].initial * math.exp((2*r + sigmaj*sigmak*rhokj)*t)
              
              beta12 += pdvCj * pdvCk * rhokj * prod * sigmaj * sigmak
              gamma12 += pdvPj * pdvPk * rhokj * prod * sigmaj * sigmak
      
      C1t=self.C1.approximate(t)
      C2t=self.C2.approximate(t)
      P1t=self.P1.approximate(t)
      P2t=self.P2.approximate(t)
      
      if (C1t*sigmat1*C2t*sigmat2==0):
        beta12=0
        gamma1 = sigmat1+sigmat2
      else:
        beta12 = self.nu**2 *beta12/(sigmat1*sigmat2*C1t*C2t)
        
        if (sigmat1**2+sigmat2**2 - 2*sigmat1*sigmat2*beta12>=0):
          gamma1 = math.sqrt(sigmat1**2+sigmat2**2 - 2*sigmat1*sigmat2*beta12)
        else:
          gamma1=0
          
      if (P1t*psit1*P2t*psit2==0):
        gamma12=0
        gamma2 = psit1+psit2
      else:
        gamma12 = self.nu**2 * gamma12/(psit1*psit2*P1t*P2t)
        
        if (psit1**2+psit2**2-2*psit1*psit2*gamma12>=0):
          gamma2 = math.sqrt(psit1**2+psit2**2-2*psit1*psit2*gamma12)
        else:
          gamma2 = 0
        
    elif (t==self.maturity):
      beta12,gamma12,gamma1,gamma2=self.getCorrelations(t,False)
    else:
      istart = round(t*self.xAmount/self.maturity)
      xArray=[(i*self.maturity)/self.xAmount for i in range(istart,self.xAmount)]
      #het kan door een numerieke fout gebeuren dat (i*self.maturity)/i niet gelijk is aan self.maturity wat voor problemen zorgt. daarom voegen we dit appart toe
      xArray.append(self.maturity)
      
      beta12 = 0
      gamma12 = 0 #we gebruiken deze waarden niet in het geval dat numeric=True
      
      try:
        gamma1=math.sqrt(integrate.simps([self.getVolatilities(x,False)[0]**2 + self.getVolatilities(x,False)[1]**2 - 2* self.getVolatilities(x,False)[0]*self.getVolatilities(x,False)[1]*self.getCorrelations(x,False)[0] for x in xArray],xArray)/(self.maturity-t))
      except ValueError: #indien de wortel negatief is...
        gamma1=0
      try:
        gamma2=math.sqrt(integrate.simps([self.getVolatilities(x,False)[2]**2 + self.getVolatilities(x,False)[3]**2 - 2* self.getVolatilities(x,False)[2]*self.getVolatilities(x,False)[3]*self.getCorrelations(x,False)[1] for x in xArray],xArray)/(self.maturity-t))
      except: #indien de wortel negatief is...
        gamma2=0
        
    self.correlationsCache[(t,numeric)] = (beta12,gamma12,gamma1,gamma2)
    return (beta12,gamma12,gamma1,gamma2)
  
  def getDPlusMin(self,t):
    if (t in self.dplusminCache):
      return self.dplusminCache[t]
    else:
      beta12,gamma12,gamma1,gamma2 = self.getCorrelations(t)
      tau = self.maturity-t
      
      C1t=self.C1.approximate(t)
      C2t=self.C2.approximate(t)
      P1t=self.P1.approximate(t)
      P2t=self.P2.approximate(t)
      
      if (C1t>0 and C2t>0 and tau>0 and gamma1>0):
        d1plus = (math.log(C1t/C2t) + tau*(gamma1**2)/2)/(gamma1*math.sqrt(tau))
        d1min = (math.log(C1t/C2t) - tau*(gamma1**2)/2)/(gamma1*math.sqrt(tau))
      elif (C2t==0 and C1t>0 and tau>0 and gamma1>0):
        d1plus = math.inf
        d1min = -math.inf
      elif (tau==0 and C1t > C2t) or gamma1==0:
        d1plus = math.inf
        d1min = math.inf
      else:
        d1plus = -math.inf
        d1min = -math.inf
      
      if (P1t>0 and P2t>0 and tau>0 and gamma2>0):
        d2plus = (math.log(P2t/P1t) + tau*(gamma2**2)/2)/(gamma2*math.sqrt(tau))
        d2min = (math.log(P2t/P1t) - tau*(gamma2**2)/2)/(gamma2*math.sqrt(tau))
      elif (P1t==0 and P2t>0 and tau>0 and gamma2>0):
        d2plus = math.inf
        d2min = -math.inf
      elif (tau==0 and P2t > P1t) or gamma2==0:  
        d2plus = math.inf
        d2min = math.inf
      else:
        d2plus = -math.inf
        d2min = -math.inf        
      
      self.dplusminCache[t] = (d1plus,d1min,d2plus,d2min)
      return (d1plus,d1min,d2plus,d2min)
  
  #partialDerivative wrt to price (Sk)
  def partialDerivative(self,t,price):
    if (price not in self.prices):
      return 0
    else:  
      if ((t,price) in self.partialCache):
        return self.partialCache[(t,price)]
      else:        
        d1plus,d1min,d2plus,d2min = self.getDPlusMin(t)
        
        E1 = self.C1.partialDerivative(t,price) *  norm.cdf(d1plus) - self.C2.partialDerivative(t,price) * norm.cdf(d1min) 
        E2 = self.P2.partialDerivative(t,price) * norm.cdf(d2plus) - self.P1.partialDerivative(t,price) * norm.cdf(d2min)
        
        self.partialCache[(t,price)] = E1 + E2
        
        return E1 + E2
        
  def approximate(self,t): 
    if (t in self.approxCache):
      return self.approxCache[t]
    elif t==self.maturity:
      return self.payoff(True)
    else:
      d1plus,d1min,d2plus,d2min = self.getDPlusMin(t)
      
      E1 = self.C1.approximate(t) * norm.cdf(d1plus) - self.C2.approximate(t) * norm.cdf(d1min)
      E2 = self.P2.approximate(t) * norm.cdf(d2plus) - self.P1.approximate(t) * norm.cdf(d2min)
      
      self.approxCache[t]=E1+E2
      return E1+E2
      
  def approxGamma(self): #op tijdstip 0
    T = self.maturity
    r = self.prices[0].drift
    
    N=len(self.prices)
    F = sum([self.thetas[i]*self.prices[i].approximate(T) for i in range(N)])
    K = sum([self.thetas[i]*self.strikes[i] for i in range(N)])
    
    M2 =  0
    for i in range(N):
      for j in range(N):
        sigmai = self.prices[i].volatility
        sigmaj = self.prices[j].volatility
              
        rhoij = self.prob.getCorrelation(self.prices[i],self.prices[j])
        
        M2+= self.thetas[i]*self.thetas[j]*self.prices[i].approximate(T)*self.prices[i].approximate(T) * math.exp(T*sigmai*sigmaj*rhoij)
    
    M2/=(F**2)
    
    alpha = (2*M2-1)/(M2-1)
    beta = 1-(1/M2)
    
    return math.exp(-r*T) * ( F * gamma.cdf(F/K,a=alpha-1,scale=beta) - K * gamma.cdf(F/K,a=alpha,scale=beta))
    
def calibrate(prob,contract,amount=100000,reltol=1e-09):
  ATM=0 #we bepalen de strike die nodig is om een At the Money optie te hebben.
  for i in range(len(contract.prices)):
    ATM+=contract.prices[i].initial*contract.thetas[i]
    
  strikes=contract.strikes #tijdelijk om later de strikes te herstellen.
  contract.setStrike(ATM)
  
  T=contract.maturity
  r=contract.prices[0].drift
  
  simulated,error = prob.simulate(contract,r,T,amount)
  
  uppernu=1
  contract.setNu(uppernu)
  
  if (contract.approximate(0)<simulated):
    while (contract.approximate(0)<simulated):
      uppernu*=2
      contract.setNu(uppernu)
    lowernu=uppernu/2
  else:
    while (contract.approximate(0)>simulated):
      uppernu/=2
      contract.setNu(uppernu)
    lowernu=uppernu
    uppernu*=2
  
  while not math.isclose(lowernu,uppernu,rel_tol=reltol):
    contract.setNu((lowernu+uppernu)/2)
    approx = contract.approximate(0)
    
    if (approx < simulated):
      lowernu=(lowernu+uppernu)/2
    else:
      uppernu=(lowernu+uppernu)/2
  
  contract.setStrike(strikes)
  return simulated,error,uppernu
  
if __name__=="__main__":
  numpy.random.seed(3141592653)
    
  thetas = [0.35,0.25,0.20,0.15,0.05]
  
  N=len(thetas)
  rho=0.5
  correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)  
  volatilities = [0.5] * N
  drifts = [0.05] * N
  initials = [100] *N
  prob = Probability(correlations,initials,drifts,volatilities)
  
  T = 1
  K = 100
  
  basket=Basket(prob,K,prob.prices,T,thetas)
  print("gamma",basket.approxGamma())
  print()
  
  print("calibrate",basket.approximate(0))
  print()
  
  amount=100000
  simulated,error = prob.simulate(basket,drifts[0],T,amount)
  
  
  print("simulated, std error",simulated,error)
  print()
  
  basket=Basket(prob,K,prob.prices,T,thetas,method="integral")
  
  print("integral",basket.approximate(0))

