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



def DE(limits,PopSize,iters):
    
    mutation_factor=0.5
    crossover_ratio=0.7
    stopping_func=None
    dim=8
  
    # solution
    s = solution()
    
    s.best = float("inf")

    # initialize population
    population = []

    population_fitness = np.array([float("inf") for _ in range(PopSize)])

    for p in range(0,PopSize):
        sol = []
        for d in range(dim):
            (lower,upper,restri)=limits[d]
            d_val = random.uniform(lower, upper)
            sol.append(d_val)

        population.append(sol)

    population = np.array(population)
    lb=[]
    ub=[]
    for pp in limits:
        lb.append(pp[0])
        ub.append(pp[1])
    # calculate fitness for all the population
    for i in range(0,PopSize):
        #fitness = objf(population[i, :])
        fitness = error_func(population[i,:])
        population_fitness[p] = fitness
        #s.func_evals += 1

        # is leader ?
        if fitness < s.best:
            s.best = fitness
            s.leader_solution = population[i, :]

    convergence_curve=np.zeros(iters)
    # start work
    print("DE is optimizing ")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    t = 0
    while t < iters:
        # should i stop
        if stopping_func is not None and stopping_func(s.best, s.leader_solution, t):
            break

        # loop through population
        for i in range(0,PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in  range(PopSize) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (population[id_2, d] - population[id_3, d])

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = np.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = error_func(mutant_sol)
            #s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < s.best:
                    s.best = mutant_fitness
                    s.leader_solution = mutant_sol
                    s.DE_solutions.append((mutant_fitness,mutant_sol))

        convergence_curve[t]=s.best
        if (t%1==0):
               print(['At iteration '+ str(t+1)+ ' the best fitness is '+ str(s.best)]);

        # increase iterations
        t = t + 1
        
        timerEnd=time.time()  
        s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime=timerEnd-timerStart
        s.convergence=convergence_curve
        s.optimizer="DE"
        

    # return solution
    return s





# get historical stock price data from yahoo finance
def get_historical_data(ticker, start_date, end_date):
    #daily_data = web.get_data_yahoo(ticker, start=start_date, end=end_date)
    #print(daily_data)
    daily_data=pd.read_csv('Data_200.csv')
    daily_data.index = pd.to_datetime(daily_data['Date'])
    #daily_data=sp500()
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

def multi_steps(ticker, start_date, end_date, max_term, min_term, prediction_term, verbose):
    global NUM_TOP_PERFORMERS
    global TIMESERIES
    global ACTUAL_VALUES
    global DATETIMES
    # get historical data
    all_data = get_historical_data(ticker, start_date, end_date)
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
     
      x=DE(limits,len(TIMESERIES),300)
      convergence[k] = x.convergence
      executionTime[k] = x.executionTime
      k+=1
      print(x.best)
      newlist=[]
      x.DE_solutions.sort()
      for i in x.DE_solutions[:3]:
        newlist.append(i[1])
      #newlist=x.leader_solution
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
MUTATION_RANGE = 0.2
NUM_TOP_PERFORMERS = 10

ticker = '^VIX'
start_date = datetime.datetime(2014, 7, 1)
end_date = datetime.datetime(2015, 4, 3)
learning_end_date = datetime.datetime(2014, 4, 3)
max_term = 90
min_term = 90
prediction_term = 2
verbose = False


#single_step(ticker, start_date, end_date, learning_end_date, max_term, min_term, generations, verbose)
multi_steps(ticker, start_date, end_date, max_term, min_term, prediction_term, verbose)
