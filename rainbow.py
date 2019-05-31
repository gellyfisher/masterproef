import math
import numpy
from scipy.stats import norm
import scipy.integrate as integrate

from prices import *
from basket import *

class Rainbow:
  def __init__(self,prob,strike,prices,maturity,nu=1,method="calibrate",xAmount=100,productmethod=False):
    self.prob = prob  
    self.strike=strike
    self.prices = prices
    self.maturity = maturity
    self.nu = nu
    
    self.method = method
    self.xAmount = xAmount
    self.productmethod = productmethod
    
    self.split()
    
    self.dplusminCache = {}  
    self.partialCache = {}
    self.approxCache = {}
    self.volatilitiesCache = {}
    self.correlationsCache = {}
    
  def payoff(self,approx=False):
    if not approx:
      temp = max([self.prices[i].value for i in range(len(self.prices))]) - self.strike
    else:
      temp = max([self.prices[i].approximate(self.maturity) for i in range(len(self.prices))]) - self.strike 
    return max(0,temp)
    
  def split(self):
    N=len(self.prices)
     
    prices1 = self.prices[:N//2] 
    prices2 = self.prices[N//2:];
    
    if (N>=4):
      self.F1=Rainbow(self.prob,0,prices1,self.maturity,self.nu,self.method,xAmount=self.xAmount,productmethod=self.productmethod)
      self.F2=Rainbow(self.prob,0,prices2,self.maturity,self.nu,self.method,xAmount=self.xAmount,productmethod=self.productmethod)
    elif (N==3):
      self.F1=prices1[0]
      self.F2=Rainbow(self.prob,0,prices2,self.maturity,self.nu,self.method,xAmount=self.xAmount,productmethod=self.productmethod)
    else:
      self.F1=prices1[0]
      self.F2=prices2[0] 
      
  def getVolatilities(self,t,numeric=True):
    if ((t,numeric) in self.volatilitiesCache):
      return self.volatilitiesCache[(t,numeric)]
    elif self.method=="calibrate" or not numeric:
      N = len(self.prices)
      r = self.prices[0].drift
      tau = self.maturity-t  
      
      sigmat = [0, 0]
      
      m = [0,N//2,N]
      F=[self.F1,self.F2]
      for i in range(2):
        for j in range(m[i],m[i+1]):
          for k in range(m[i],m[i+1]):
            sigmaj = self.prices[j].volatility
            sigmak = self.prices[k].volatility
            rhokj = self.prob.getCorrelation(self.prices[j],self.prices[k])
            
            pdvFj = F[i].partialDerivative(t,self.prices[j])
            pdvFk = F[i].partialDerivative(t,self.prices[k])
            
            if not self.productmethod:
              Sj = self.prices[j].approximate(t)
              Sk = self.prices[k].approximate(t)
              
              sigmat[i] += pdvFj * pdvFk * rhokj * Sk * Sj * sigmaj * sigmak
            else:
              prod = self.prob.prices[0].initial * self.prob.prices[1].initial * math.exp((2*r + sigmaj*sigmak*rhokj)*t)  
              sigmat[i] += pdvFj * pdvFk * rhokj * prod * sigmaj * sigmak
            
            
      F1t=self.F1.approximate(t)
      F2t=self.F2.approximate(t)
      
      sigmat1 = 0 if F1t==0 else self.nu * math.sqrt(sigmat[0]) / F1t
      sigmat2 = 0 if F2t==0 else self.nu * math.sqrt(sigmat[1]) / F2t
      
    elif (t==self.maturity):
       sigmat1,sigmat2=self.getVolatilities(t,False)
    else:
      istart = int(round(t*self.xAmount/self.maturity))
      xArray=[(i*self.maturity)/self.xAmount for i in range(istart,self.xAmount)]
      #het kan door een numerieke fout gebeuren dat (i*self.maturity)/i niet gelijk is aan self.maturity wat voor problemen zorgt. daarom voegen we dit appart toe
      xArray.append(self.maturity)
      
      sigmat1=math.sqrt(integrate.simps([self.getVolatilities(x,False)[0]**2 for x in xArray],xArray)/(self.maturity-t))
      sigmat2=math.sqrt(integrate.simps([self.getVolatilities(x,False)[1]**2 for x in xArray],xArray)/(self.maturity-t))
       
    self.volatilitiesCache[(t,numeric)]=(sigmat1,sigmat2)
    return (sigmat1,sigmat2)

  def getCorrelations(self,t,numeric=True):
    if ((t,numeric) in self.correlationsCache):
      return self.correlationsCache[(t,numeric)]
    elif self.method=="calibrate" or not numeric:
      sigmat1,sigmat2 = self.getVolatilities(t) 
      N=len(self.prices)
      r=self.prices[0].drift
        
      delta12 = 0
      for j in range(0,N//2):
        for k in range(N//2,N):
          
          sigmaj = self.prices[j].volatility
          sigmak = self.prices[k].volatility
          rhokj = self.prob.getCorrelation(self.prices[j],self.prices[k])
    
          pdvFj = self.F1.partialDerivative(t,self.prices[j])
          pdvFk = self.F2.partialDerivative(t,self.prices[k])
          
          if not self.productmethod:
            Sj = self.prices[j].approximate(t)
            Sk = self.prices[k].approximate(t)
          
            delta12 += pdvFj * pdvFk * rhokj * Sk * Sj * sigmaj * sigmak
          else:
            prod = self.prob.prices[0].initial * self.prob.prices[1].initial * math.exp((2*r + sigmaj*sigmak*rhokj)*t)  
              
            delta12 += pdvFj * pdvFk * rhokj * prod * sigmaj * sigmak
      
      F1t=self.F1.approximate(t)
      F2t=self.F2.approximate(t)
      
      if (sigmat1==0 or sigmat2==0):
        delta12=0
        gammax = sigmat1+sigmat2
      else:
        delta12 = self.nu**2 *delta12/(sigmat1*sigmat2*F1t*F2t)
        gammax = math.sqrt(sigmat1**2+sigmat2**2 - 2*sigmat1*sigmat2*delta12)
       
    elif (t==self.maturity):
       delta12,gammax=self.getCorrelations(t,False)
    else:
      istart = int(round(t*self.xAmount/self.maturity))
      xArray=[(i*self.maturity)/self.xAmount for i in range(istart,self.xAmount)]
      #het kan door een numerieke fout gebeuren dat (i*self.maturity)/i niet gelijk is aan self.maturity wat voor problemen zorgt. daarom voegen we dit appart toe
      xArray.append(self.maturity)
      
      delta12 = 0 #wordt niet meer gebruikt in dit geval
      try:
        gammax = math.sqrt(integrate.simps([self.getVolatilities(x,False)[0]**2 + self.getVolatilities(x,False)[1]**2 - 2* self.getVolatilities(x,False)[0]*self.getVolatilities(x,False)[1]*self.getCorrelations(x,False)[0] for x in xArray],xArray)/(self.maturity-t))
      except:
        gammax = 0 #indien we wortel van een negatief getal kregen.
        
    self.correlationsCache[(t,numeric)] = (delta12,gammax) 
    return (delta12,gammax)
    
  def getDPlusMin(self,t):  
    if (t in self.dplusminCache):
      return self.dplusminCache[t]
    else:
      _,gammax = self.getCorrelations(t)
      tau = self.maturity-t
      
      F1t=self.F1.approximate(t)
      F2t=self.F2.approximate(t)
      
      if (F1t>0 and F2t>0 and tau>0):
        dplus = (math.log(F1t/F2t) + tau*(gammax**2)/2)/(gammax*math.sqrt(tau))
        dmin = (math.log(F1t/F2t) - tau*(gammax**2)/2)/(gammax*math.sqrt(tau))        
      elif (F2t==0 and F1t>0 and tau>0):
        dplus = math.inf
        dmin = -math.inf
      elif tau==0 and F1t > F2t:
        dplus = math.inf
        dmin = math.inf
      else:
        dplus = -math.inf
        dmin = -math.inf
        
      self.dplusminCache[t] = (dplus,dmin)
      return (dplus,dmin) 
        
  def approximate(self,t):
    if (t in self.approxCache):
      return self.approxCache[t]
    elif t==self.maturity:
      return self.payoff(True)
    elif self.strike==0:
      dplus,dmin = self.getDPlusMin(t)
    
      self.approxCache[t]=self.F1.approximate(t) * norm.cdf(dplus) + self.F2.approximate(t) * (1-norm.cdf(dmin))
      return self.approxCache[t]
      
    else:
      N=len(self.prices)
    
      _,sigmat2=self.getVolatilities(t)
      sigmatx=0
      
      dplus,dmin=self.getDPlusMin(t)
      #we bepalen de partiele afgeleide van X adhv 2.4.7
      pdvX=[]
      for j in range(0,N):
        pdvX.append(self.F1.partialDerivative(t,self.prices[j]) * norm.cdf(dplus) - self.F2.partialDerivative(t,self.prices[j]) * norm.cdf(dmin))
      
      #sigmatx uit lemma 2.4.4
      for j in range(0,N):
        for k in range(0,N):
          sigmaj = self.prices[j].volatility
          sigmak = self.prices[k].volatility
          rhokj = self.prob.getCorrelation(self.prices[j],self.prices[k])
          
          if not self.productmethod:
            Sj = self.prices[j].approximate(t)
            Sk = self.prices[k].approximate(t)
            
            sigmatx += pdvX[j] * pdvX[k] * rhokj * Sk * Sj * sigmaj * sigmak
          else:
            prod = self.prob.prices[0].initial * self.prob.prices[1].initial * math.exp((2*r + sigmaj*sigmak*rhokj)*t)  
            
            sigmatx += pdvX[j] * pdvX[k] * rhokj * prod * sigmaj * sigmak
      
      #Uit stelling 2.4.6
      Xt = self.F1.approximate(t) * norm.cdf(dplus) - self.F2.approximate(t) * norm.cdf(dmin)
      if Xt!=0:
        sigmatx = math.sqrt(sigmatx)/Xt
      else:
        sigmatx = 0
      
      #delta_x2 uit lemma 2.4.5 
      deltax2=0
      for j in range(0,N):
        for k in range(N//2,N):
          sigmaj = self.prices[j].volatility
          sigmak = self.prices[k].volatility
          rhokj = self.prob.getCorrelation(self.prices[j],self.prices[k])
          
          if not self.productmethod:
            Sj = self.prices[j].approximate(t)
            Sk = self.prices[k].approximate(t)
          
            deltax2 += pdvX[j] * self.F2.partialDerivative(t,self.prices[k]) * rhokj * Sk * Sj * sigmaj * sigmak
          else:
            prod = self.prob.prices[0].initial * self.prob.prices[1].initial * math.exp((2*r + sigmaj*sigmak*rhokj)*t)  
            
            deltax2 += pdvX[j] * self.F2.partialDerivative(t,self.prices[k]) * rhokj * prod * sigmaj * sigmak
      
      if (sigmatx!=0 and sigmat2!=0):
        deltax2 = deltax2/(sigmat2 * sigmatx * Xt * self.F2.approximate(t))
      else:
        deltax2=0
      
      correlations = numpy.ones((2,2))*deltax2+numpy.diag([1-deltax2]*2)  
      volatilities = [sigmat2,sigmatx]
      drifts = [self.prices[0].drift] * 2
      initials = [self.F2.approximate(0), Xt]
      
      prob = Probability(correlations,initials,drifts,volatilities)
      
      basket=Basket(prob,self.strike,prob.prices,self.maturity,[1,1],nu=self.nu,method=self.method,xAmount=self.xAmount,productmethod=self.productmethod)
    
      self.approxCache[t]=basket.approximate(t)
      return self.approxCache[t]
    
  def partialDerivative(self,t,price):
    #dit moeten we in feite enkel kunnen berekenen indien K=0
    if (price not in self.prices):
      return 0
    else: 
      if ((t,price) in self.partialCache):
        return self.partialCache[(t,price)]
      else:  
        dplus,dmin = self.getDPlusMin(t)
        
        self.partialCache[(t,price)]=self.F1.partialDerivative(t,price) *  norm.cdf(dplus) + self.F2.partialDerivative(t,price) * (1-norm.cdf(dmin))
        return self.partialCache[(t,price)]
        
if __name__=="__main__":
  numpy.random.seed(3141592653)
  
  N=5
  rho=0.5
  correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)  
  volatilities = [0.3,0.4,0.5,0.5,0.7]
  drifts = [0.05] * N
  initials = [90,92,94,96,98,100]
  prob = Probability(correlations,initials,drifts,volatilities)
  
  T = 1
  K = 100
  
  rainbow=Rainbow(prob,K,prob.prices,T,method="integral")
  
  amount=100000
  simulated,error = prob.simulate(rainbow,drifts[0],T,amount)
    
  print("simulated, std error",simulated,error)
  print()
  
  print(rainbow.approximate(0))
  print()  