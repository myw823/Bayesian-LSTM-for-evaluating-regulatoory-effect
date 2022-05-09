import pandas as pd
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from pickle import dump

# Random 80/10/10 split
def data_split(df:pd.DataFrame):
    X = df.drop(columns = ['PM2.5']).copy()
    y = df['PM2.5']
    
    # split into a 80% train : 20% others
    X_train, X_testAndvalid, y_train, y_testAndvalid = train_test_split(X, y, test_size=0.2)

    # split the 20% others into 10% test and 10% valid
    X_test, X_valid, y_test, y_valid = train_test_split(X_testAndvalid, y_testAndvalid, test_size=0.5)
    
    print(X_train.shape), print(y_train.shape)
    print(X_valid.shape), print(y_valid.shape)
    print(X_test.shape), print(y_test.shape)

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def standardize(X_train, X_test, X_valid, num_split):
    # Standardize training set
    scaler = StandardScaler()
    columns = ["TEMP","VISIB","WDSP","RH","wdir","pres","population_density","number_of_vehicles"]
    X_train[columns] = scaler.fit_transform(X_train[columns]) 

    # Standardize test and validation set
    X_test[columns] = scaler.transform(X_test[columns]) 
    X_valid[columns] = scaler.transform(X_valid[columns]) 

    # Save the scalar for evaluating performance
    path = f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/scaler{num_split}.pkl"
    
    with open(path,'wb') as file:
        dump(scaler, file)

    return X_train, X_test, X_valid


def iterative_imputate(X_train, X_test, X_valid):
    # Construct the imputer based on training set
    iterative_imputer = IterativeImputer(estimator=ExtraTreesRegressor())
    columns = ["TEMP","VISIB","WDSP","RH","wdir","pres"]
    X_train[columns] = iterative_imputer.fit_transform(X_train[columns])

    # Use the imputer on test and valid
    X_test[columns] = iterative_imputer.transform(X_test[columns])
    X_valid[columns] = iterative_imputer.transform(X_valid[columns])


    return X_train, X_test, X_valid


def data_preprocess():
    df_air_quality = pd.read_csv("~/Documents/FYP-causal/datasets/Ready-data/air-quality-data-daily-averaged.csv")
    df_meteorology = pd.read_csv("~/Documents/FYP-causal/datasets/Ready-data/city_UK_meteorology_data_2010-2020.csv")
    df_wdir_And_pres = pd.read_csv("~/Documents/FYP-causal/datasets/Ready-data/heathrow_combined.csv")
    
    # Combining air meteorology data into a new df (group by date)
    dfAirAndMeteo = df_meteorology
    dfAirAndMeteo['PM2.5'] = df_air_quality['Value']
    dfAirAndMeteo['wdir'] = df_wdir_And_pres['wdir']
    dfAirAndMeteo['pres'] = df_wdir_And_pres['pres']

    dfAirAndMeteo = dfAirAndMeteo[['date','TEMP','VISIB','WDSP','RH','PM2.5','wdir','pres']]
    dfAirAndMeteo['date'] = pd.to_datetime(dfAirAndMeteo['date'])
    dfAirAndMeteo.set_index('date', inplace=True)
    dfAirAndMeteo.to_csv("~/Documents/FYP-causal/bayesian-self/air_and_meteo.csv")

    # regulation data to time-serires binary vector
    dfRegulation = pd.DataFrame(pd.date_range(start=datetime(2010,1,1), end=datetime(2020,1,31)), columns=["date"])

    df_regulation_status = pd.read_csv("~/Documents/FYP-causal/datasets/Ready-data/London_regulations.csv")
    for index, row in df_regulation_status.iterrows():
        # fill 0s to the column first
        dfRegulation[row['policy_name']] = 0
        
        dfRegulation.loc[(dfRegulation['date'] >= row['start_date']) & (dfRegulation['date'] <= row['end_date']), row['policy_name']] = 1
    
    dfRegulation.set_index("date", inplace=True)
    dfRegulation.to_csv("~/Documents/FYP-causal/bayesian-self/regulationStatuses.csv")

    # Combining Air, Meteo and regulation data
    df_combined = dfAirAndMeteo.join(dfRegulation)

    # Socio-economic data
    df_population = pd.read_csv("~/Documents/FYP-causal/datasets/Ready-data/population_density.csv")
    df_vehicles = pd.read_csv("~/Documents/FYP-causal/datasets/Ready-data/number_or_vehicles.csv")
    dfSocEcon = pd.DataFrame(pd.date_range(start=datetime(2010,1,1), end=datetime(2020,1,31)), columns=["date"])

    dfSocEcon['population_density'] = dfSocEcon['date'].apply(lambda date: df_population[df_population['year'] == date.year]['population_density'].values[0])
    dfSocEcon['number_of_vehicles'] = dfSocEcon['date'].apply(lambda date: df_vehicles[df_vehicles['year'] == date.year]['number'].values[0])
    
    dfSocEcon.set_index("date", inplace=True)
    dfSocEcon.to_csv("~/Documents/FYP-causal/bayesian-self/socio-Econ.csv")

    # Combine three datasets into data.csv
    df_all = df_combined.join(dfSocEcon)

    # Add year, month, and dayOfWeek as categorical features
    df_all['year'] = df_all.index.year
    df_all['month'] = df_all.index.month
    df_all['dayOfWeek'] = df_all.index.dayofweek

    df_all.to_csv("~/Documents/FYP-causal/bayesian-self/data.csv")

    # TODO: Repeated five times
    for n in range(1,6):
        # 80/10/10 data split
        X_train, y_train, X_test, y_test, X_valid, y_valid = data_split(df_all)

        # Filling missing data with iterative imputation
        X_train, X_test, X_valid = iterative_imputate(X_train, X_test, X_valid)

        # Standardization to mean and std
        X_train, X_test, X_valid = standardize(X_train, X_test, X_valid, n)
        
        all_inputData = pd.concat([X_train, X_test, X_valid])
        all_outputData = pd.concat([y_train, y_test, y_valid])

        all_inputData.sort_values(by="date", inplace=True)
        all_inputData.sort_values(by="date", inplace=True)

        all_inputData.to_csv("~/Documents/FYP-causal/bayesian-self/IOdataAll/inputData.csv")
        all_outputData.to_csv("~/Documents/FYP-causal/bayesian-self/IOdataAll/outputData.csv")


        X_train.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{n}/training-input.csv")
        y_train.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{n}/training-output.csv")
        X_test.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{n}/test-input.csv")
        y_test.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{n}/test-output.csv")
        X_valid.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{n}/valid-input.csv")
        y_valid.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{n}/valid-output.csv")


def main():
    data_preprocess()


if __name__ == "__main__":
    main()