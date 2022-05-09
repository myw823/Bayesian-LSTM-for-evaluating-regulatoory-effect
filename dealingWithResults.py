import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt


def calculate_monthly_mean(factual, counterfactual) -> pd.DataFrame:
    monthlyMean_df = pd.DataFrame()
    
    # index -> DatetimeIndex
    factual['date'] = pd.to_datetime(factual['date'])
    counterfactual['date'] = pd.to_datetime(counterfactual['date'])

    # drop irrevalent columns
    counterfactual = counterfactual[['date','mean']]
    print(factual.resample('M', on='date').mean()['PM2.5'])

    monthlyMean_df['observed'] = factual.resample('M', on='date')['PM2.5'].mean()
    monthlyMean_df['counterfactual'] = counterfactual.resample('M', on='date')['mean'].mean()
    print(monthlyMean_df)

    monthlyMean_df['difference'] = monthlyMean_df.counterfactual.sub(monthlyMean_df.observed, axis=0)
    monthlyMean_df['percentage_difference'] = monthlyMean_df.apply(lambda row: (row['difference'] / row['observed']) * 100, axis=1)


    return monthlyMean_df




def main():
    results = pd.read_csv("~/Documents/FYP-causal/bayesian-self/results/CounterfactualPM2.5-mean.csv")

    # Get all columns names
    columns = results.drop(columns=['mean','date','factualPM2.5','Unnamed: 0']).columns
    
    results[columns] = results[columns] + 5
    results['mean'] = results[columns].mean(axis=1) # re-compute mean
    results.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
    results.drop(["a"], axis=1, inplace=True)

    results.to_csv("~/Documents/FYP-causal/bayesian-self/results/results.csv")

    observed_y = pd.read_csv("~/Documents/FYP-causal/bayesian-self/IOdataAll/outputData.csv")

    # Monthly
    monthly_mean_df = calculate_monthly_mean(observed_y, results)

    # Plotting monthly
    ax = monthly_mean_df['percentage_difference'].plot(marker='o', markersize=3.)
    monthly_mean_df.to_csv('~/Documents/FYP-causal/bayesian-self/results/monthly_mean_results.csv')
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative PM2.5 reduction by air regulations (%)')
    plt.show()

if __name__ == "__main__":
    main()