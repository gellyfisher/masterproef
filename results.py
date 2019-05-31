import math
import numpy
from scipy.stats import norm

import time

from prices import *
from basket import *
from rainbow import *

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

def tabel12():
  numpy.random.seed(3141592653)
  amount=500000 #aantal simulaties
  N=5
  thetas=[0.35,0.25,0.20,0.15,0.05]
  rhoArray = [0.1,0.5]
  volatilitiesArray = [[0.2] * N, [0.5] * N]
  driftsArray = [[0.05] * N,[0.1] * N]
  initials = [100] * N
  TArray=[1,3]
  KArray=[100,90,110]
  
  result=""
  nus={}
  for T in TArray:
    for K in KArray:
      for drifts in driftsArray:
        for volatilities in volatilitiesArray:
          for rho in rhoArray:
            correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)
            prob=Probability(correlations,initials,drifts,volatilities)
            
            basket=Basket(prob,K,prob.prices,T,thetas)  
            
            if (K==100):
              simulated,error,nu=calibrate(prob,basket,amount,1e-09)
              nus[(T,drifts[0],volatilities[0],rho)]=nu
            else:
              nu=nus[(T,drifts[0],volatilities[0],rho)]
              basket.setNu(nu)
            
              simulated,error = prob.simulate(basket,drifts[0],T,amount)
            
            approx=basket.approximate(0)
            result+=("%d & %.2f & %.1f & %.1f & %6.4f & %.4f & %.4f & %.4f & %.4f \\\\\n")%(K,drifts[0],volatilities[0],rho, simulated,error,approx,abs(simulated-approx)/simulated,nu)
      result+="\\hline\n"
    result+="\n" #zodat we makkelijk T=1 en T=3 uit elkaar kunnen halen

  print(result)
  
  
def tabel345():  
  numpy.random.seed(3141592653)
  amount=500000 #aantal simulaties
  N=5
  thetas=[0.35,0.25,0.20,0.15,0.05]
  rhoArray = [0.1,0.5]
  volatilitiesArray = [[0.2] * N, [0.5] * N]
  driftsArray = [[0.05] * N,[0.1] * N]
  initials = [100] * N
  TArray=[1,3]
  KArray=[100,90,110]
  
  sum=0
  sumalt=0
  
  result="" #benaderingen zelf
  result2="" #absolute fouten
  result3="" #tijden
  for T in TArray:
    for K in KArray:
      for drifts in driftsArray:
        for volatilities in volatilitiesArray:
          for rho in rhoArray:
            correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)
            prob=Probability(correlations,initials,drifts,volatilities)
            
            basket=Basket(prob,K,prob.prices,T,thetas,method="integral",productmethod=True)  
            basket2=Basket(prob,K,prob.prices,T,thetas,nu=1,method="calibrate")
            
            start = time.time_ns()
            simulated,error = prob.simulate(basket,drifts[0],T,amount)
            simulatedTime = (time.time_ns() - start) / (10**6) #in ms
            
            start = time.time_ns()
            gamma = basket.approxGamma()
            gammaTime = (time.time_ns() - start) / (10**6) #in ms
            
            start = time.time_ns()
            approx = basket.approximate(0)
            approxTime = (time.time_ns() - start)  / (10**6) #in ms
            
            start = time.time_ns()
            approx2 = basket2.approximate(0)
            approx2Time = (time.time_ns() - start)  / (10**6) #in ms
            
            errors=[abs(a-simulated) for a in [approx2,approx,gamma]]
            best=min(errors)
            
            errors2=[abs(a-simulated) for a in [approx2,approx]]
            best2=min(errors2)
            
            format = "%d & %d & %.2f & %.1f & %.1f & %6.4f & %.4f &"
            for i in range(3):
              if errors[i]==best:
                format+=" \\bfseries %.4f "
              else:
                format+=" %.4f "
              if i!=2:
                if (errors2[i]==best2):
                  format+=" \\textsuperscript{*} "
                format+="&"
            format+="\\\\\n"
            
            sum+=abs(approx2-simulated)
            sumalt+=abs(approx-simulated)
            
            result+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2, approx, gamma)
            result2+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2-simulated, approx-simulated, gamma-simulated)
            result3+=("%d & %d & %.2f & %.1f & %.1f & %.2f \\text{ ms} & %.2f \\text{ ms} & %.2f \\text{ ms} & %.2f \\text{ ms} \\\\\n")%(T,K,drifts[0],volatilities[0],rho, simulatedTime, approx2Time, approxTime, gammaTime)
            
      result+="\\hline\n"
      result2+="\\hline\n"
      result3+="\\hline\n"
            
  print(result)
  print()
  print()
  print(result2)
  print()
  print()
  print(result3)
  print()
  print()
  print(sum)
  print(sumalt)
  
  
def figuur2(): #we genereren random basket opties.
  numpy.random.seed(3141592653)
  
  amount=100000 #aantal simulaties
  N=5
  
  sum=0
  sumalt=0
  
  res=[]
  res2=[]
  
  for i in range(48):
    thetas = numpy.random.rand(N)
  
    temp=numpy.random.rand(N,N)
    correlations=0.5*(temp+temp.transpose()) #maak de matrix symmetrisch
    numpy.fill_diagonal(correlations,N); #zorg dat de matrix positief definiets is
    correlations = correlations/N #herschaal zodat het een correlatiematrix word.
    
    volatilities = numpy.random.randn(N)
    drifts = [numpy.random.uniform(0,0.5)]*N
    initials = numpy.random.rand(N)*100
    T = numpy.random.uniform(0.1,3)
    
    K = numpy.random.uniform(0,10)
    
    prob=Probability(correlations,initials,drifts,volatilities)
  
    basket=Basket(prob,K,prob.prices,T,thetas,method="integral")  
    basket2=Basket(prob,K,prob.prices,T,thetas,nu=1,method="calibrate")
    
    simulated,error = prob.simulate(basket,drifts[0],T,amount)
    
    approx = basket.approximate(0)
    approx2 = basket2.approximate(0)
    
    res.append(approx/simulated)
    res2.append(approx2/simulated)
    
    # print(approx,approx2,simulated)
    # print("SUMS",sum,sumalt)
    # print()
    
  df=pd.DataFrame()
  df['benadering']=res2
  df['alt. benadering']=res

  fig, ax = plt.subplots(2,1,sharex=True)

  bp1=sns.boxplot(df['benadering'],ax=ax[0])
  bp2=sns.boxplot(df['alt. benadering'],ax=ax[1])
  
  plt.setp(bp1.artists, edgecolor = 'k', facecolor='sandybrown')
  plt.setp(bp2.artists, edgecolor = 'k', facecolor='sandybrown')
  plt.setp(bp1.lines, color='k')
  plt.setp(bp2.lines, color='k')

  ax[0].tick_params(axis='x',labelbottom=True)

  ax[0].set_yticks([])
  ax[1].set_yticks([])

  plt.tight_layout()
  plt.savefig("../randombasketboxplots.png")
  
def tabel67(): #rainbow resultaten
  numpy.random.seed(3141592653)
  amount=500000 #aantal simulaties
  N=5
  
  rhoArray = [0.1,0.5]
  volatilitiesArray = [[0.2] * N, [0.5] * N]
  driftsArray = [[0.05] * N,[0.1] * N]
  initials = [100] * N
  TArray=[1,3]
  KArray=[100,90,110]
  
  sum=0
  sumalt=0
  
  result="" #benaderingen zelf
  result2="" #absolute fouten
  for T in TArray:
    for K in KArray:
      for drifts in driftsArray:
        for volatilities in volatilitiesArray:
          for rho in rhoArray:
            correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)
            prob=Probability(correlations,initials,drifts,volatilities)
            
            rainbow=Rainbow(prob,K,prob.prices,T,method="integral")
            rainbow2=Rainbow(prob,K,prob.prices,T,method="calibrate")            
            
            simulated,error = prob.simulate(rainbow,drifts[0],T,amount)
           
            approx = rainbow.approximate(0)
            approx2 = rainbow2.approximate(0)
            
            
            
            errors=[abs(a-simulated) for a in [approx2,approx]]
            best=min(errors)
            
            format = "%d & %d & %.2f & %.1f & %.1f & %6.4f & %.4f &"
            for i in range(2):
              if errors[i]==best:
                format+=" \\bfseries %.4f "
              else:
                format+=" %.4f "
              if i!=1:
                format+="&"
            format+="\\\\\n"
  
            result+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2, approx)
            result2+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2-simulated, approx-simulated)
            
      result+="\\hline\n"
      result2+="\\hline\n"
            
  print(result)
  print()
  print()
  print(result2)
  
def figuur3(): #we genereren random rainbow opties.
  numpy.random.seed(3141592653)
  
  amount=100000 #aantal simulaties
  N=5
  
  sum=0
  sumalt=0
  res=[]
  res2=[]
  
  for i in range(48): 
    thetas = numpy.random.randn(N)
  
    temp=numpy.random.rand(N,N)
    correlations=0.5*(temp+temp.transpose()) #maak de matrix symmetrisch
    numpy.fill_diagonal(correlations,N); #zorg dat de matrix positief definiet is
    correlations = correlations/N #herschaal zodat het een correlatiematrix word.
    
    volatilities = numpy.random.randn(N)
    drifts = [numpy.random.uniform(0,0.5)]*N
    initials = numpy.random.rand(N)*100
    T = numpy.random.uniform(0.1,3)
    
    K = numpy.random.uniform(0,10)

    
    prob=Probability(correlations,initials,drifts,volatilities)
  
    rainbow=Rainbow(prob,K,prob.prices,T,method="integral")  
    rainbow2=Rainbow(prob,K,prob.prices,T,nu=1,method="calibrate")
    
    simulated,error = prob.simulate(rainbow,drifts[0],T,amount)
    
    approx = rainbow.approximate(0)
    approx2 = rainbow2.approximate(0)
    
    res.append(approx/simulated)
    res2.append(approx2/simulated)
    
  df=pd.DataFrame()
  df['benadering']=res2
  df['alt. benadering']=res

  fig, ax = plt.subplots(2,1,sharex=True)

  bp1=sns.boxplot(df['benadering'],ax=ax[0])
  bp2=sns.boxplot(df['alt. benadering'],ax=ax[1])
  
  plt.setp(bp1.artists, edgecolor = 'k', facecolor='sandybrown')
  plt.setp(bp2.artists, edgecolor = 'k', facecolor='sandybrown')
  plt.setp(bp1.lines, color='k')
  plt.setp(bp2.lines, color='k')

  ax[0].tick_params(axis='x',labelbottom=True)

  ax[0].set_yticks([])
  ax[1].set_yticks([])

  plt.tight_layout()
  plt.savefig("../randomrainbowboxplots.png")
    
  print(sum)
  print(sumalt)

def tabel89():  
  numpy.random.seed(3141592653)
  amount=500000 #aantal simulaties
  N=5
  thetas=[0.35,0.25,0.20,0.15,0.05]
  rhoArray = [0.1,0.5]
  volatilitiesArray = [[0.2] * N, [0.5] * N]
  driftsArray = [[0.05] * N,[0.1] * N]
  initials = [100] * N
  TArray=[1,3]
  KArray=[100,90,110]
  
  sum=0
  sumalt=0
  
  result="" #benaderingen zelf
  result2="" #absolute fouten
  
  for T in TArray:
    for K in KArray:
      for drifts in driftsArray:
        for volatilities in volatilitiesArray:
          for rho in rhoArray:
            temp=numpy.random.randn(N,N)
            correlations=0.5*(temp+temp.transpose()) #maak de matrix symmetrisch
            numpy.fill_diagonal(correlations,N); #zorg dat de matrix positief definiets is
            correlations = correlations/N #herschaal zodat het een correlatiematrix word.
            # correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)
              
            prob=Probability(correlations,initials,drifts,volatilities)
            
            basket2=Basket(prob,K,prob.prices,T,thetas,method="integral",productmethod=False)  
            basket=Basket(prob,K,prob.prices,T,thetas,nu=1,method="integral",productmethod=True)
            
            simulated,error = prob.simulate(basket,drifts[0],T,amount)
            
            approx = basket.approximate(0)
            approx2 = basket2.approximate(0)
            
            sum+=approx2-simulated
            sumalt+=approx-simulated
            
            errors=[abs(a-simulated) for a in [approx2,approx]]
            best=min(errors)
            
            format = "%d & %d & %.2f & %.1f & %.1f & %6.4f & %.4f &"
            for i in range(2):
              if errors[i]==best:
                format+=" \\bfseries %.4f "
              else:
                format+=" %.4f "
              if i!=1:
                format+="&"
            format+="\\\\\n"
  
            result+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2, approx)
            result2+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2-simulated, approx-simulated)
            
      result+="\\hline\n"
      result2+="\\hline\n"
            
  print(result)
  print()
  print()
  print(result2)
  print()
  print()
  print(sum)
  print(sumalt)
  
  
def figuur1():  
  numpy.random.seed(3141592653)
  amount=500000 #aantal simulaties
  N=5
  thetas=[0.35,0.25,0.20,0.15,0.05]
  rhoArray = [0.1,0.5]
  volatilitiesArray = [[0.2] * N, [0.5] * N]
  driftsArray = [[0.05] * N,[0.1] * N]
  initials = [100] * N
  TArray=[1,3]
  KArray=[100,90,110]
  
  sum=0
  sumalt=0
  
  res=[]
  res2=[]
  
  result="" #benaderingen zelf
  result2="" #absolute fouten
  for T in TArray:
    for K in KArray:
      for drifts in driftsArray:
        for volatilities in volatilitiesArray:
          for rho in rhoArray:
            correlations = numpy.ones((N,N))*rho+numpy.diag([1-rho]*N)
            prob=Probability(correlations,initials,drifts,volatilities)
            
            basket=Basket(prob,K,prob.prices,T,thetas,method="integral",productmethod=False)  
            basket2=Basket(prob,K,prob.prices,T,thetas,nu=1,method="calibrate")
            
            simulated,error = prob.simulate(basket,drifts[0],T,amount)
            
            approx = basket.approximate(0)
            approx2 = basket2.approximate(0)
            
            sum+=approx2/simulated
            sumalt+=approx/simulated
            
            res.append(approx/simulated)
            res2.append(approx2/simulated)
            
            errors=[abs(1-a/simulated) for a in [approx2,approx]]
            best=min(errors)
            
            format = "%d & %d & %.2f & %.1f & %.1f & %6.4f & %.4f &"
            for i in range(2):
              if errors[i]==best:
                format+=" \\bfseries %.4f "
              else:
                format+=" %.4f "
              if i!=1:
                format+="&"
            format+="\\\\\n"
  
            result+=format%(T,K,drifts[0],volatilities[0],rho, simulated, error, approx2/simulated, approx/simulated)
            
      result+="\\hline\n"
            
  print(result)
  print()
  print()
  print(sum/48) #0.9418153089164248
  print(sumalt/48) #1.0774573386623003
  print()  
  
  df=pd.DataFrame()
  df['benadering']=res2
  df['alt. benadering']=res

  fig, ax = plt.subplots(2,1,sharex=True)

  bp1=sns.boxplot(df['benadering'],ax=ax[0])
  bp2=sns.boxplot(df['alt. benadering'],ax=ax[1])

  
  plt.setp(bp1.artists, edgecolor = 'k', facecolor='sandybrown')
  plt.setp(bp2.artists, edgecolor = 'k', facecolor='sandybrown')
  plt.setp(bp1.lines, color='k')
  plt.setp(bp2.lines, color='k')

  ax[0].tick_params(axis='x',labelbottom=True)

  ax[0].set_yticks([])
  ax[1].set_yticks([])

  plt.tight_layout()
  plt.savefig("../boxplots.png")
  
if __name__=="__main__":
  # tabel123()
  # tabel45()
  figuur2()
  # tabel67()
  # figuur3()
  # tabel89() #vergelijk tussen productmethode en geen productmethode
  # figuur1() 
  