#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import datetime
import numpy as np
import scipy.optimize as opt
import pandas_datareader as web
#import pandas.io.data as web
import matplotlib.pyplot as plt
import pkg_resources
import pandas as pd
from solution import solution
import time

# LPPL function
def fit_func(t, parameters):
    # Log-Periodic Power Law (LPPL) function (JLS model function)
    # as for a, b, tc, m, c, w and phi, refer to the papers for the detail.
    # start_time is to decide dt, the term window size of the learning data,
    # not used in this function.
    (a, b, tc, m, c, w, phi, start_time) = parameters
    tm = np.power(tc - t, m)
    return np.exp(a + b*tm + c*tm*np.cos(w*np.log(tc-t)-phi))

# Error function for scipy.optimize.fmin_tnc
def error_func(parameters):
    # TIMESERIES and ACTUAL_VALUES are global variables used for the learning process.
    # they will be overwritten when executing stepwize calculations.
    global MAX_ERROR
    global TIMESERIES
    global ACTUAL_VALUES
    # start_time is to decide dt, the term window size of the learning data.
    # start_time can be fixed by limiting the range of it by [0, 0, true].
    (a, b, tc, m, c, w, phi, start_time) = parameters
    if math.isnan(start_time):
      return MAX_ERROR
    timeseries = TIMESERIES[int(start_time):]
    actual_values = ACTUAL_VALUES[int(start_time):]
    # calculate the mean squared errors of the estimated values
    # the error is measured on the actual_values so that can be used
    estimated_values = [fit_func(t, parameters) for t in timeseries]
    diff = np.divide(np.subtract(estimated_values, actual_values), actual_values)
    mse = np.sum(np.power(diff, 2))/(len(timeseries)-1)
    return mse

def GWO(limits,SearchAgents_no,Max_iter):
    
    
    #Max_iter=1000
    #lb=-100
    #ub=100
    #dim=30  
    #SearchAgents_no=5
    dim=8
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    omega=[]
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")
    
    """if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim"""
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        (lower,upper,restri)=limits[i]
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no) * (upper - lower) + lower
    
    Convergence_curve=np.zeros(Max_iter)
    s=solution()

     # Loop counter
    print("GWO is optimizing ")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                (lower,upper,restri)=limits[j]
                Positions[i,j]=np.clip(Positions[i,j], lower, upper)

            # Calculate objective function for each search agent
            #fitness=objf(i,Positions[i,:])
            fitness = error_func(Positions[i,:])
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=Alpha_score  # Update beta
                Beta_pos=Alpha_pos.copy()
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score):                 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
                omega.append((Delta_score,Delta_pos))
        
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        
        Convergence_curve[l]=Alpha_score;

        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="GWO"
    #s.objfname=objf.__name__
    s.bestIndividual=[Alpha_pos,Beta_pos,Delta_pos]
    omega.sort()
    for p in omega[0:3]:
         s.Omega.append(p[1])
    return s
        


# get historical stock price data from yahoo finance
def get_historical_data(ticker):
    #daily_data = web.get_data_yahoo(ticker, start=start_date, end=end_date)
    #print(daily_data)
    daily_data=pd.read_csv('Data_200.csv')
    #daily_data=pd.read_csv('../Data/data.csv')
    daily_data.index = pd.to_datetime(daily_data['Date'])
    num_days = len(daily_data)
    print(num_days)
    timeseries = range(0, num_days)
    values = [daily_data['Close'][i] for i in xrange(num_days)]
    datetimes = map(lambda tm: datetime.datetime(tm.year, tm.month, tm.day), daily_data.index.tolist())
    return [timeseries, values, datetimes]

# pick up the target data from the all historical data series
def get_learning_data(all_data, learning_end_date, max_term):
    (timeseries, actual_values, datetimes) = all_data
    learning_end_pos = 0
    for dt in datetimes:
      if dt >= learning_end_date:
        break
      learning_end_pos += 1
    learning_start_pos = max(0, learning_end_pos - max_term)
    return (timeseries[learning_start_pos:learning_end_pos],
            actual_values[learning_start_pos:learning_end_pos],
            datetimes[learning_start_pos:learning_end_pos])


def estimate(timeseries,parameters):
        return [fit_func(t, parameters) for t in timeseries]

def multi_steps(ticker, max_term, min_term, prediction_term, verbose):
    global NUM_TOP_PERFORMERS
    global TIMESERIES
    global ACTUAL_VALUES
    global DATETIMES
    # get historical data
    all_data = get_historical_data(ticker)
    (timeseries_all, actual_values_all, datetimes_all) = all_data
    # execute multiple steps
    p = False
    results = []
    #for k in range (0,3):
    k=0
    #print(len(datetimes_all))
    #NumOfRuns=(len(datetimes_all)-max_term)/max_term
    NumOfRuns=90
    convergence = [0]*NumOfRuns
    executionTime = [0]*NumOfRuns
    for learning_end_date in datetimes_all[max_term:(max_term+NumOfRuns)]:
      # get learning data for single step execution
      LEARNING_DATA = get_learning_data(all_data, learning_end_date, max_term)
      (TIMESERIES, ACTUAL_VALUES, DATETIMES) = LEARNING_DATA
      init_a = [1.0, 5.0, False]
      init_b = [0.1, 2.0, False]
      init_tc =[TIMESERIES[-1], TIMESERIES[-1]+250, False]
      init_m = [0.0, 1.0, True]
      init_c = [-1.0, 1.0, False]
      init_w = [0.1, 2.0, False]
      init_phi = [0.0, np.pi, False]
      init_start = [0, TIMESERIES[-1]-min_term, True]
      limits = (init_a, init_b, init_tc, init_m, init_c, init_w, init_phi, init_start)
      x=GWO(limits,len(TIMESERIES),300)
      convergence[k] = x.convergence
      executionTime[k] = x.executionTime
      k+=1
      print(x.bestIndividual)
      newlist=x.bestIndividual
      #newlist.extend(x.Omega)
      results.append((TIMESERIES[-1], newlist))
    # draw chart
    draw_multi_steps(results, all_data, prediction_term)

def draw_multi_steps(results, all_data, prediction_term):
    (timeseries_all, actual_values_all, datetimes_all) = all_data
    # plot the actual data
    plt.scatter(timeseries_all, actual_values_all, color='black')
    # get band range of the predictions
    best = []
    upper = []
    lower = []
    critical_time = {}
    for (pos, xs) in results:
      r = []
      for x in xs:
        # get and record the prediction
        r.append(estimate([pos+prediction_term],x))
        # get and record the critical time
        ct = x[2]
        if critical_time.has_key(ct):
          critical_time[ct] += 1
        else:
          critical_time[ct] = 1
      if len(xs) > 0:
        # get the best guess
        e = estimate([pos+prediction_term],xs[0])
        best.append((pos, e))
        # get the upper/lower guesses (excluding the extream ones)
        if len(xs) > 3:
          r.sort()
          upper.append((pos, r[len(r)-1-1]))
          lower.append((pos, r[1]))
        else:
          upper.append((pos, e))
          lower.append((pos, e))
    # 
      c = 0
    #for prediction in (best, upper, lower):
    for prediction in (upper,lower):
      ts = [pos+prediction_term for (pos, e) in prediction]
      pv = [e for (pos, e) in prediction]
      #plt.plot(ts, pv, linewidth=(3 if c==0 else 1))
      plt.plot(ts, pv)
      c += 1
    plt.show()
    # draw critical time distribution
    print (critical_time)
    keys = critical_time.keys()
    keys.sort()
    vs = []
    for k in keys:
      vs.append(critical_time[k])
    plt.bar(keys, vs)
    plt.show()


MAX_ERROR = 10.0
NUM_TOP_PERFORMERS = 10
start_date = datetime.datetime(2014, 7, 1)
end_date = datetime.datetime(2015, 4, 3)
learning_end_date = datetime.datetime(2014, 4, 3)
ticker = '^VIX'
learning_end_date = datetime.datetime(2014, 4, 3)
max_term = 90
min_term = 90
prediction_term = 2
verbose = False



multi_steps(ticker, max_term, min_term, prediction_term, verbose)
