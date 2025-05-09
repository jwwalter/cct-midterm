#modules
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import os
import matplotlib.pyplot as plt

###  LOAD DATA
#read data as pandas dataframe
df = pd.read_csv('~/project/cct-midterm/data/plant_knowledge.csv')
df = df.drop(columns='Informant') #drop informant ID column
print(df)
#change to numpy array & reshape
data = df.values
N,M = data.shape

###  MODEL

#name model
cct_model = pm.Model()

#define model
with cct_model:
    #prior for D (competence) - vector of size N (# informants) 
    #D = pm.Beta('D', alpha=2, beta=2, shape=N)
    D = pm.Uniform('D', lower=0, upper=1, shape=N)
    #prior for Z (answer) - vector of size M (# items)
    Z = pm.Bernoulli('Z',p=0.5, shape=M)
    #reshape D
    D_rs = D[:,None]
    #probability equation
    prob = Z*D_rs+(1-Z)*(1-D_rs)
    #likelihood equation - using my updated data
    Lh = pm.Bernoulli('Lh',p=prob, observed=data)
    #take sample
    #-----problem here????
    idata_cct_model = pm.sample(draws=2000, chains=4, tune=2500, target_accept=0.95)

###   INFERENCE
#summary of posterior
az.summary(idata_cct_model)

#trace plots
az.plot_trace(idata_cct_model)

#save inferences
idata_cct_model.to_netcdf("cct_model_results.nc")

#saving/viewing plots - USED CHATGPT FOR THESE FIRST LINES
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(FIG_DIR, exist_ok=True)

#save trace plot
def save_trace_plot(idata, model_name="cct_model"):
    
    #get variable names, except not the likelihood
    var_names = [v for v in idata.posterior.data_vars if v != "Lh"]

    #make trace plot
    az.plot_trace(idata, var_names=var_names)
    plt.suptitle(f"{model_name} Trace Plot", fontsize=16)
    plt.tight_layout()
    trace_plot_path = os.path.join(FIG_DIR, f"{model_name}_trace_plot.png")
    plt.savefig(trace_plot_path)
    plt.close()
save_trace_plot(idata_cct_model, model_name="cct_model")
#convergence diagnostics


### ESTIMATE INFORMANT COMPETENCE

#take posterior samples of D
D_post = idata_cct_model.posterior['D']
#calc posterior mean for D
D_mean = D_post.mean(dim=["chain", "draw"])
print(D_mean)

#make & save plot
az.plot_posterior(idata_cct_model, var_names=['D'], hdi_prob=0.95)
plt.suptitle(f"Posterior Distributions of Informant Competence (D)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "informant_competence_posterior.png"))
plt.close()

#-----ISSUES HERE!!
#find best & worst informants
most_competent = D_mean.argmax(dim='informant')
least_competent = D_mean.argmin(dim='informant')
print(most_competent)
print(least_competent)

### ESTIMATE CONSENSUS ANSWERS

# #take posterior samples of Z
Z_post = idata_cct_model.posterior['Z']
#calc posterior mean for Z
Z_mean = Z_post.mean(dim=["chain", "draw"])
print(Z_mean)

# #round mean to get most likely consensus answer 
Z_most_likely = np.round(Z_mean)
print(Z_most_likely)

# #make & save plot of consensus answers
az.plot_posterior(idata_cct_model, var_names=['Z'], hdi_prob=0.95)
plt.suptitle(f"Posterior Distributions of Consensus Answers (Z)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "consensus_answers_posterior.png"))
plt.close()