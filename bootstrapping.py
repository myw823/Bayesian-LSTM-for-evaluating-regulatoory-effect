import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def bootstrapping(ate):
    ate = ate.transpose()
    ate.to_csv("~/Documents/FYP-causal/bayesian-self/results/ate_transpose.csv")
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    size=10000
    func = np.mean
    # Create an empty array to store replicates
    replicates = np.empty(size)

    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        sample = np.random.choice(50,size=50)
        sample = ate.iloc[sample].values
        # Get bootstrap replicate and append to bs_replicates
        replicates[i] = func(sample)
    
    return replicates, 

def plot(bs_replicates):
    # Plot the PDF for bootstrap replicates as histogram
    plt.hist(bs_replicates,bins=30, rwidth=0.9)

    # Showing the related percentiles
    plt.axvline(x=np.percentile(bs_replicates,[2.5]), ymin=0, ymax=1,label='2.5th percentile',c='y')
    plt.axvline(x=np.percentile(bs_replicates,[97.5]), ymin=0, ymax=1,label='97.5th percentile',c='r')

    plt.xlabel("Average Treatment Effect (ATE) μg/m3")
    plt.ylabel("Density")
    plt.title("Probability Density Function")
    plt.legend()
    plt.show()

def main():
    results = pd.read_csv("~/Documents/FYP-causal/bayesian-self/results/results.csv")
    columns = results.drop(columns=['mean','date','factualPM2.5','Unnamed: 0']).columns
    
    ate = results[columns].sub(results['factualPM2.5'],axis=0)
    ate.rename(columns=lambda column_name: 'ate-'+column_name[0:3], inplace=True)
    ate.to_csv("~/Documents/FYP-causal/bayesian-self/results/ate.csv")
    bs_replicates, threeDArr = bootstrapping(ate) # array of (1000 resample arrays of (50 random runs))
    
    # Plotting
    plot(bs_replicates)

    conf_interval = np.percentile(bs_replicates,[2.5,97.5])
    print(f"mean:  {np.mean(bs_replicates)}")
    print(conf_interval)

    print(f"\n {threeDArr}")

if __name__ == "__main__":
    main()