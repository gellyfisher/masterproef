import math
import numpy

def choleskyDecomposition(matrix):
  ret=numpy.zeros(matrix.shape,dtype='float64')
  for i in range(0,matrix.shape[0]):
    for j in range(0,i):
      ret[i,j]=(matrix[i,j]-sum([ret[i,k]*ret[j,k] for k in range(0,j)]))/ret[j,j]
  
    ret[i,i]=math.sqrt(matrix[i,i]-sum([ret[i,k]*ret[i,k] for k in range(0,i)]))
  
  return ret;

class Probability:
  def __init__(self,correlations,initials,drifts,volatilities,constant=True):
    self.correlations = correlations
    self.G = choleskyDecomposition(correlations)
    
    self.motions=[]
    self.prices=[]   
    
    self.antithetic = False   # Om bij te houden dat of we nieuwe data moeten genereren of we gewoon -self.data kunnen gebruiken.

    for i in range(correlations.shape[0]):
      self.motions.append(Motion(self))
      self.prices.append(Price(initials[i],drifts[i],volatilities[i],self.motions[i]))
      
    self.constant=constant
    
    self.getCorrelationMatrix={} #gebruiken we om sneller getCorrelation te berekenen
    for i in range(len(self.prices)):
      for j in range(len(self.prices)):
        self.getCorrelationMatrix[(self.prices[i],self.prices[j])] = self.correlations[i,j]
    
  def getCorrelation(self,price1,price2):
    return self.getCorrelationMatrix[(price1,price2)]
    # return self.correlations[self.prices.index(price1),self.prices.index(price2)]
  
  #sampled de prijzen zelf
  def samplePrices(self,T):
    if (self.constant):
      self.sampleMotions(T)
      for price in self.prices:
        price.value=price.initial * math.exp(
              price.volatility*price.motion.getSampled()
              +(price.drift-0.5*price.volatility**2)*T
            )
    else:
      intervals=100 #aantal intervallen waarover we de niet constante sigma beschouwen
      
      for price in self.prices:
        price.value=price.initial
        
      t=0
      for i in range(intervals):
        self.sampleMotions(T/intervals,False)
        for price in self.prices:
          price.value=price.value * math.exp(
                price.volatility(t)*price.motion.getSampled()
                +(price.drift-0.5*price.volatility(t)**2)*(T/intervals)
              )
        t+=T/intervals
    
    return [price.value for price in self.prices];
   
  #sampled de Brownse bewegingen
  def sampleMotions(self,t,antithetic=True):
    n=len(self.motions)
    
    if (self.antithetic and antithetic):
      self.data = -self.data
      self.antithetic=False
    else:
      self.data = numpy.random.randn(n,1)*math.sqrt(t);  
      self.antithetic = True
      
    correlatedData = self.G.dot(self.data);
    
    for i in range(0,n):
      self.motions[i].setSampled(correlatedData[i,0])
      
  def simulate(self,contract,r,T,amount=100000):
    total=0
    total2=0
    for _ in range(amount): 
      self.samplePrices(T)
      temp = math.exp(-r*T) * contract.payoff()
      total += temp
      total2 += temp**2
        
    simulated=total/amount
    variance=total2/amount-simulated**2
    error=math.sqrt(variance/amount)
    
    return (simulated,error)

#implementeert de functionaliteiten van een Brownse beweging
class Motion:
  def __init__(self,prob):
    self.prob = prob
  
  def setSampled(self,value):
    self.sampled=value
    
  def getSampled(self):
    return self.sampled;
    
#implementeert de functionaliteiten van een geometrische Brownse beweging
class Price:
  def __init__(self, initial, drift, volatility, motion):
    self.initial = initial
    self.drift = drift
    self.volatility = abs(volatility) #we gaan er van uit dat onze volatiliteit altijd positief is.
    self.motion = motion
    self.value = initial
    
  def partialDerivative(self,t,price):
    if price==self:
      return 1
    else: 
      return 0
  
  def approximate(self,t):
    if (t>0):
      return self.initial * math.exp(self.drift*t)
    else:
      return self.initial
  
  