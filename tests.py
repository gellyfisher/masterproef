from prices import *
from basket import *
from exchange import *
from option import *
from rainbow import *

def testCholesky():
  correlations = numpy.matrix('1 0.8 0.6 0.2; 0.8 1 0.55 0.65; 0.6 0.55 1 0.57; 0.2 0.65 0.57 1',dtype='float64')
  ret=choleskyDecomposition(correlations);
  
  assert(numpy.isclose(ret.dot(ret.transpose()),correlations,1e-09).all()) #er is geen exacte gelijkheid door afrondingfoutjes.
  
  n=19
  temp=numpy.random.rand(n,n)
  correlations=0.5*(temp+temp.transpose()) #maak de matrix symmetrisch
  numpy.fill_diagonal(correlations,n); #zorg ervoor dat de matrix zeker positief definiet is.
  ret=choleskyDecomposition(correlations);
  
  assert(numpy.isclose(ret.dot(ret.transpose()),correlations,1e-09).all()) #er is geen exacte gelijkheid door afrondingfoutjes.
  
def testCorrelated():
  n = 4
  amount = 100000 #aantal iteraties
  correlations = numpy.matrix('1 0.8 0.6 0.2; 0.8 1 0.55 0.65; 0.6 0.55 1 0.57; 0.2 0.65 0.57 1',dtype='float64')
  G = choleskyDecomposition(correlations)
  
  for n1 in range(0,n):
    for n2 in range(n1,n):
      total=0 #we schatten de correlatie tussen W_n1 en W_n2
      
      for _ in range(0,amount):
        data = numpy.random.randn(n,1);  
        correlatedData = G.dot(data);
        total+=correlatedData[n1,0]*correlatedData[n2,0]
      
      #de relatieve tolerantie mag niet te hoog zijn omdat de gesampelde verwachtingswaarde nooit exact zal zijn.
      assert(math.isclose(total/amount,correlations[n1,n2],rel_tol=0.05))
  
def testCallOption():
  correlations = numpy.matrix('1',dtype='float64')
  volatilities = [0.30]
  drifts = [0.04]
  initials = [75]
  prob=Probability(correlations,initials,drifts,volatilities)
  
  T=1
  
  call=Option(60,prob.prices[0],T)
  
  amount=100000
  simulated,error = prob.simulate(call,drifts[0],T,amount)
    
  assert(math.isclose(total/amount,call.approximate(0),rel_tol=0.01))
  
def testCEOs():
  correlations = numpy.matrix('1 0.5; 0.5 1',dtype='float64')
  amount=1000000
  
  diffs=[]
  
  volitiltiesArray= [[0.20]*2,[0.50]*2]
  driftsArray = [[0.04]*2,[0]*2]
  initialsArray = [[100,80],[100,90],[100,100]]
  TArray = [0.08,0.5]
  KArray = [60,80,100,120]
  
  for volatilities in volitiltiesArray:
    for drifts in driftsArray:
      for initials in initialsArray:
        for K in KArray:
          for T in TArray:
            prob=Probability(correlations,initials,drifts,volatilities)
  
            call1=Option(K,prob.prices[0],T)
            call2=Option(K,prob.prices[1],T)
            ceo=CEO(prob,call1,call2)
  
            simulated,error = prob.simulate(ceo,drifts[0],T,amount)
              
            approx=ceo.approximate(0)
            
            print(volatilities,drifts,initials,K,T)
            print(total/amount)
            print(approx)
            print(approx-(total/amount))
            diffs.append(approx-(total/amount))
            print()
            print()
            
def testCEO():
  correlations = numpy.matrix('1 0.5; 0.5 1',dtype='float64')
  amount=1000000
  
  volatilities = [0.20]*2
  drifts = [0.04]*2
  initials = [100,80]
  T = 0.08
  K = 60
  
  prob=Probability(correlations,initials,drifts,volatilities)
  
  call1=Option(K,prob.prices[0],T)
  call2=Option(K,prob.prices[1],T)
  ceo=CEO(prob,call1,call2)

  simulated,error = prob.simulate(ceo,drifts[0],T,amount)
    
  approx=ceo.approximate(0)
  
  print(volatilities,drifts,initials,K,T)
  print(total/amount)
  print(approx)
  print(ceo.approximate2(0))
  print(ceo.approximate3(0))
  print()
  
def testCEO2():
  correlations = numpy.matrix('1 0.5; 0.5 1',dtype='float64')
  amount=100000
  
  volatilities = [0.50]*2
  drifts = [0.05]*2
  initials = [100,100]
  T = 1
  K = 50
  
  prob=Probability(correlations,initials,drifts,volatilities)
  
  call1=Option(K,prob.prices[0],T)
  call2=Option(-K,prob.prices[1],T)
  ceo1=CEO(prob,call1,call2)
  
  call4=Option(-K,prob.prices[1],T,-1)
  call3=Option(K,prob.prices[0],T,-1)
  ceo2=CEO(prob,call4,call3)
  basket=Basket(prob,2*K,prob.prices,T,[1,-1]) 

  print(ceo1.approximate(0)+ceo2.approximate(0))
  print(basket.approximate(0))
    
  totalCeo = 0
  totalBasket = 0;
  for _ in range(amount): 
    prob.samplePrices(T)
    totalCeo+=math.exp(-drifts[0]*T)*(ceo1.payoff()+ceo2.payoff())
    totalBasket+=math.exp(-drifts[0]*T)*basket.payoff()
  
  print(totalCeo/amount)
  print(totalBasket/amount)
  print()
            
def testExchange():
  correlations = numpy.matrix('1 0.5; 0.5 1',dtype='float64')
  volatilities = [0.30,0.25]
  drifts = [0.04]*2
  initials = [75,65]
  prob=Probability(correlations,initials,drifts,volatilities)
  
  T=1
  
  xchg=Exchange(prob,prob.prices[0],prob.prices[1],T)
  
  amount=2000000
  simulated,error = prob.simulate(xchg,drifts[0],T,amount)
    
  print(total/amount)
  print(xchg.approximate(0))
  
def testExchangeAsBasket():
  N=2
  correlations = numpy.ones((N,N))*0.5+numpy.diag([0.5]*N)
  volatilities = [0.2] * N
  drifts = [0.05] * N
  initials = [100] *N
  prob=Probability(correlations,initials,drifts,volatilities)
  
  T=1
  
  exchg = Exchange(prob,prob.prices[0],prob.prices[1],T)
  basket=Basket(prob,0,prob.prices,T,[1,-1],method="integral")  #exchange optie
  
  amount=100000
  print(exchg.approximate(0),*prob.simulate(exchg,drifts[0],T,amount))  
  print(basket.approximate(0),*prob.simulate(basket,drifts[0],T,amount))
  
def nonConstantTest():
  correlations = numpy.matrix('1',dtype='float64')
  volatilities = [lambda t:  math.sqrt(0.18*t)]
  
  drifts = [0.04]
  initials = [75]
  prob=Probability(correlations,initials,drifts,volatilities,False)
  
  T=1
  
  call=Option(60,prob.prices[0],T)
  
  amount=100000
  simulated,error = prob.simulate(call,drifts[0],T,amount)
    
  # assert(math.isclose(total/amount,call.approximate(0),rel_tol=0.01))
  print(total/amount,call.approximate(0))
  
def testBasket():
  N=3
  correlations = numpy.ones((N,N))*0.5+numpy.diag([0.5]*N)
  volatilities = [0.2] * N
  drifts = [0.05] * N
  initials = [100] *N
  prob=Probability(correlations,initials,drifts,volatilities)
  
  T=1
  K=100
  
  basket=Basket(prob,K,prob.prices,T,[1,1,-1],1e20)  #exchange optie
    
  print(basket.approximate(0))
  
def testProductLogNormals():
  # rho12=0.5
  # rho13=0.4
  # rho23=0.2
  rho=0.5
  correlations = numpy.ones((2,2))*rho+numpy.diag([1-rho]*2)
  volatilities = [0.2,0.4]
  drifts = [0.05] * 2
  initials = [100] * 2
  prob=Probability(correlations,initials,drifts,volatilities)
  
  r=drifts[0]
  T=1
  sigma1=volatilities[0]
  sigma2=volatilities[1]
  
  amount=200000
  total=0
  for _ in range(amount): 
    prob.samplePrices(T)
    total+=prob.prices[0].value*prob.prices[1].value
    
  print("simulated",total/amount)
  
  mu = (2*r -0.5*sigma1**2 - 0.5*sigma2**2)*T
  sigma = math.sqrt((sigma1**2 + sigma2**2 + 2 * rho * sigma1 * sigma2)*T)
  
  print("expectation",prob.prices[0].initial*prob.prices[1].initial * math.exp(mu+0.5*sigma**2))
  
  print("naief",prob.prices[0].approximate(T)*prob.prices[1].approximate(T))
  
if __name__=="__main__":
  numpy.random.seed(3141592653)
  
  # testCholesky()
  # testCorrelated()
  # testCallOption()
  # testExchange()
  # testCEO2()
  # testCEOs()
  testExchangeAsBasket()
  # nonConstantTest()
  # testBasket()
  # testProductLogNormals()