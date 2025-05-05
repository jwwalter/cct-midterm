#modules
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

#load data - read data as pandas dataframe
df = pd.read_csv('~/project/cct-midterm/data/plant_knowledge.csv')
df = df.drop(columns='Informant') #drop informant ID column
print(df)
#change to numpy array & reshape
data = df.values
N,M = data.shape

#model
cct_model = pm.model()

with cct_model:
    #prior for Di (prob of informant giving correct ans/competence) 
        #di = pm.Normal('di',mu=0, sigma=1)
    #prior for Zj (correct ans for an item)
        #zj = pm.Bernoulli('zj',p=0.5)
    #prior for D (competence) - vector of size N (# informants) 
    D = pm.Beta('D', alpha=1, beta=1, shape=N)
    #prior for Z (answer) - vector of size M (# items)
    Z = pm.Bernoulli('Z',p=0.5, shape=M)
    #prob of informant giving correct answer
    #pij = zj*di+(1-zj)*(1-di)
    D_rs = D[:,None]
    prob = Z*D_rs+(1-Z)*(1-D_rs)
    #likelihood
    Lh = pm.Bernoulli('Lh',p=prob, observed=data)
    #take sample
    idata_cct_model = pm.sample(draws=2000, chains=4, nuts_sampler='pymc')

#inference

#convergence diagnostics

#estimate informant competence

#estimate consensus answers
