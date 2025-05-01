#modules
import pandas as pd
import pymc as pm
import arviz as az

#load data - read data as pandas dataframe
df = pd.read_csv('~/project/cct-midterm/data/plant_knowledge.csv')
df = df.drop(columns='Informant') #drop informant ID column
print(df)

#define priors

#model
pm.model()

#inference

#convergence diagnostics

#estimate informant competence

#estimate consensus answers
