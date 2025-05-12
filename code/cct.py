
#import modules
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import mode
from pathlib import Path

#Used ChatGPT to fix file path & plot saving issues with this part
# Set base path to project root regardless of where script is run
BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

### LOAD THE DATA
#read csv into pandas dataframe
df = pd.read_csv(BASE_DIR / 'data' / 'plant_knowledge.csv')
df = df.drop(columns='Informant') #drop informant ID column
print(df)
#change to numpy array & reshape
data = df.values
N,M = data.shape

### BUILD MODEL & DEFINE PRIORS
#build the cct model
with pm.Model() as cct_model:
    #prior for D (competence) - vector of size N (# informants)
    D = pm.Uniform('D', lower=0.5, upper=1.0, shape=N)
    #prior for Z (answer) - vector of size M (# items)
    Z = pm.Bernoulli('Z', p=0.5, shape=M)

    #reshape D 
    D_reshaped = D[:, None]

    #probability equation 
    prob = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
    #likelihood equation, using reshaped data
    Lh = pm.Bernoulli('Lh', p=prob, observed=data)
    
### PERFORM INFERENCE
    #take sample
    trace = pm.sample(draws=2000, chains=4, tune=1000, target_accept=0.9)

#check convergence diagnostics
summary = az.summary(trace, var_names=['D', 'Z'])
print(summary)

#visualize posterior dist for competence
az.plot_posterior(trace, var_names=['D'])
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'competence_posteriors.png')

### ESTIMATE INFORMANT COMPETENCE
#find most and least competent informants
competences = trace.posterior['D'].mean(dim=('chain','draw')).values
most_competent = np.argmax(competences)
least_competent = np.argmin(competences)
print(f'most competent informant: {most_competent}, competence value: {competences[most_competent]:.2f}')
print(f'least competent informant: {least_competent}, competence value: {competences[least_competent]:.2f}')

### ESTIMATE CONSENSUS ANSWERS
#posterior mean prob for each answer consensus (Z)
consensus_probs = trace.posterior['Z'].mean(dim=('chain','draw')).values
print('posterior mean consensus probabilities:')
print(consensus_probs)

#most likely consensus answer key, using rounding
consensus_key = np.round(consensus_probs).astype(int)
print('consensus key from model:')
print(consensus_key)

#plot for posterior of consensus
az.plot_posterior(trace, var_names=['Z'])
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'consensus_posteriors.png')

#compare with naive majority vote
majority = mode(data, axis=0, keepdims=False).mode
print('naive majority vote answer key:')
print(majority)