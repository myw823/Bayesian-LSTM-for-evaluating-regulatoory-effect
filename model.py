import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pickle import load
from sklearn.model_selection import GridSearchCV
from PIL import ImageFont

import tensorflow as tf
import tensorflow_probability as tfp
import edward2 as ed

from keras.models import Sequential, save_model, load_model
from keras.layers import LSTM, Dense, Embedding, Flatten, Dropout, Concatenate, RNN, Reshape
from keras.utils.vis_utils import plot_model
from keras.losses import mean_squared_error, binary_crossentropy, kl_divergence, MeanSquaredError
from keras import Input, activations, Model, optimizers
import keras.backend as K

    

def evaluate_model(model, valid_Xt, valid_I, valid_y, num_split, batch_size):
    input_Xt = valid_Xt[:, 0:8, :]
    input_It = valid_I
    input_time = valid_Xt[:, 8:10, 0:1]

	# forecast dataset
    air_pred, regulation_pred = model.predict(x=[input_Xt, input_time, input_It], batch_size=batch_size)
    #output = model.predict(x=[input_Xt, input_month,input_dayOfWeek, input_It], batch_size=batch_size)
    mse = MeanSquaredError()
    value = mse(valid_y, air_pred).numpy()
    print(value)
    print()
    print(regulation_pred)
    return value

def tune_hyper(trainData_list, validData_list):
    X_train_Xt, X_train_I, y_train = trainData_list
    X_valid_Xt, X_valid_I, y_valid = validData_list

    results = pd.DataFrame()

    input_valid_Xt = X_valid_Xt[:, 0:8, :]
    input_valid_It = X_valid_I
    input_valid_time = X_valid_Xt[:, 8:10, 0:1]

    '''
    reg_embed_dims = [3,5]
    num_hidden_units = [128,256]
    batch_sizes = [32,64]
    '''
    grid = [[3,128,32], [3,128,64], [3,256,32], [3,256,64], [5,128,32], [5,128,64], [5,256,32], [5,256,64]]
    for i in range(len(grid)):
        K.clear_session()
        combination = grid[i]
        scores = []
        model = build_model(X_train_Xt, X_train_I, combination[0], combination[1], combination[2])
        print(f"\nStart testing Model with {combination[0]}   {combination[1]}   {combination[2]}\n")
        for _ in range(10):
            model = build_model(X_train_Xt, X_train_I, combination[0], combination[1], combination[2])
            trained_model = train_model(model, X_train_Xt, X_train_I, y_train, combination[2])
            air_pred, _ = trained_model.predict(x=[input_valid_Xt, input_valid_time, input_valid_It], batch_size=combination[2])
            mse_value = mean_squared_error(y_valid, air_pred).numpy().mean()
            scores.append(mse_value)
            print("\n scores updated:")
            print(scores)
            print("\n")
        results[f"{combination[0]},{combination[1]},{combination[2]}"] = scores
        results.to_csv("~/Documents/FYP-causal/bayesian-self/IOdata1/tuning_hyper.csv")
        #if i in [1,3,5,7]:
        #    results.boxplot()
        #        print(results.describe())
        #    plt.savefig("~/Documents/FYP-causal/visualization/parameters_tuning.png")
        #    results = pd.DataFrame()
    results.to_csv("~/Documents/FYP-causal/bayesian-self/IOdata1/tuning_hyper.csv")
    # Plotting
    print(results.describe())
    results.boxplot()
    plt.savefig(f"~/Documents/FYP-causal/visualization/tuning_hyper.png")

def combineTVT(num_split):

    X_train = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/training-input.csv")
    X_test = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/test-input.csv")
    X_valid = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/valid-input.csv")
    y_train = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/training-output.csv")
    y_test = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/test-output.csv")
    y_valid = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/valid-output.csv")
    X_train['date'] = pd.to_datetime(X_train['date'])
    X_test['date'] = pd.to_datetime(X_test['date'])
    X_valid['date'] = pd.to_datetime(X_valid['date'])

    all_inputData = pd.concat([X_train, X_test, X_valid])
    all_outputData = pd.concat([y_train, y_test, y_valid])
    #all_inputData['date'] = all_inputData['date'].str.strip()
    #all_inputData['date'] = pd.to_datetime(all_inputData['date'])
    print(all_inputData.date)
    all_inputData.sort_values(by="date", inplace=True)
    all_outputData.sort_values(by="date", inplace=True)

    all_inputData.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/all-Input.csv")
    all_outputData.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/all-Output.csv")



def compute_ate(trainData_list, validData_list, testData_list): # in Xt, I, y
    reg_embed_dim = 5
    num_hidden_unit = 128
    batch_size = 32

    for num_split in [1,2,3,4,5]:
        all_inputData = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/all-Input.csv")
        all_outputData = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/all-Output.csv").drop(columns=['date'])


        all_inputData['date'] = pd.to_datetime(all_inputData['date'])
        all_inputData.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
        all_inputData.drop(["a"], axis=1, inplace=True)
        all_outputData.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
        all_outputData.drop(["a"], axis=1, inplace=True)

        XAndt = looking_back(all_inputData[["date","TEMP","VISIB","WDSP","RH","wdir","pres","population_density","number_of_vehicles","month","dayOfWeek"]], 7)
        i = all_inputData.drop(["date","TEMP","VISIB","WDSP","RH","wdir","pres","year","month","dayOfWeek","population_density","number_of_vehicles"], axis=1)
        i = i.values
        y = all_outputData.values
        
        train_no_reg_I = np.zeros(shape=i.shape)
        
        x = XAndt[:, 0:8, :]
        time = XAndt[:, 8:10, 0:1]

        results = pd.DataFrame()
        for i in range(10):
            K.clear_session()
            model = build_model(trainData_list[0], trainData_list[1], reg_embed_dim, num_hidden_unit, batch_size)
            #train_model
            trained_model = train_model(model, trainData_list[0], trainData_list[1], trainData_list[2], batch_size)
            
            # Counterfactual outcome
            air_pred, _ = trained_model.predict(x=[x, time, train_no_reg_I], batch_size=batch_size)
            results[f"{num_split}-{i}run"] = air_pred.flatten()
            results.to_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/counterfactual.csv")






# X_train_Xt shape:  (2946, 10, 6)
def train_model(model:Model, X_train_Xt, X_train_I, output_y, batch_size, plot_loss=False):
    input_Xt = X_train_Xt[:, 0:8, :]
    input_It = X_train_I
    input_time = X_train_Xt[:, 8:10, 0:1]


    metrics = []
    model.compile(loss={'air_prediction': 'mse', 'regulation_prediction': 'binary_crossentropy'}, optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
    history = model.fit(x=[input_Xt, input_time, input_It], y=[output_y, X_train_I], epochs=30, batch_size=batch_size)

    if plot_loss:
        plt.plot(history.history['air_prediction_loss'])
        plt.title('Mean Squared Error loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('~/Documents/FYP-causal/visualization/training_MSE_loss_visualization.png', bbox_inches='tight')
        plt.clf()

        plt.plot(history.history['regulation_prediction_loss'])
        plt.title('Binary Cross Entropy loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('~/Documents/FYP-causal/visualization/training_BCE_loss_visualization.png', bbox_inches='tight')

    return model

def build_model(X_train_Xt:np.ndarray, X_train_I:np.ndarray, regulation_embedding_dim=5, num_hidden_units=128, batch_size=32, plot=False):
    
    input_Xt = Input(shape=(8,8), name='ProxyData')
    input_time = Input(shape=(2,), name='Time')
    input_It = Input(shape=(32,), name='Regulation statuses')

    # KL loss function
    KL_divergence_fn = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /  tf.cast(len(X_train_Xt), dtype=tf.float32))


    # Building model

    # RNN(LSTM)
    ht = RNN(ed.layers.LSTMCellFlipout(num_hidden_units, batch_input_shape=(batch_size, 8, 8), kernel_regularizer=ed.regularizers.NormalKLDivergence(scale_factor=1./X_train_Xt.shape[0]), recurrent_regularizer=ed.regularizers.NormalKLDivergence(scale_factor=1./X_train_Xt.shape[0]),), name='Bayesian-LSTM')(input_Xt)

    ht = Dropout(0.2)(ht)

    # Embedding
    e1t = ed.layers.EmbeddingReparameterization(84, 3, embeddings_regularizer = ed.regularizers.NormalKLDivergence(scale_factor=1./X_train_Xt.shape[0]), name='Time_embedding')(input_time) # Embedding time

    e2t = ed.layers.EmbeddingReparameterization(64, regulation_embedding_dim, embeddings_regularizer = ed.regularizers.NormalKLDivergence(scale_factor=1./X_train_Xt.shape[0]), name='Regulation_embedding')(input_It) # Embedding regulation statuses
    
    e1t = Flatten()(e1t)
    e2t = Flatten()(e2t)

    input_for_pred = Concatenate(1)([ht, e1t, e2t])

    air_pred = tfp.layers.DenseFlipout(1, kernel_divergence_fn=KL_divergence_fn, activation=None, name='air_prediction')(input_for_pred)
    regulation_pred = Dense(32, activation='sigmoid', name='regulation_pred')(tfp.layers.DenseFlipout(32, kernel_divergence_fn=KL_divergence_fn, activation=None, name='propensity_score')(input_for_pred))

    model = Model([input_Xt, input_time, input_It], [air_pred, regulation_pred])

    if plot:
        print(model.summary())
        
        plot_model(model, "~/Documents/FYP-causal/visualization/Bayesian-LSTM_model_visualization.png")

    return model


# Each window = number of lagged observations(0or7) + current day
def looking_back(data_df:pd.DataFrame, look_back:int):
    allInputData = pd.read_csv("~/Documents/FYP-causal/bayesian-self/IOdataAll/inputData.csv")
    allInputData = allInputData[["date","TEMP","VISIB","WDSP","RH","wdir","pres","population_density","number_of_vehicles"]]
    allInputData['date'] = pd.to_datetime(allInputData['date'])

    dataX = []
    for date in data_df['date']:

        # Handle not enough 7 days at past
        if date <= datetime(2010,1,7):
            toAppend_df = allInputData[allInputData['date'] <= date].sort_values(by="date")
        else:
            # Past 7 days
            sevenDaysAgo = date - timedelta(days=look_back)
            toAppend_df = allInputData[(allInputData['date'] <= date) & (allInputData['date'] >= sevenDaysAgo)].sort_values(by="date")

        toAppend_df.drop('date', axis=1, inplace=True)
        toAppend_list = toAppend_df.values.tolist() # is a 2-D List

        # fill 0s
        while len(toAppend_list) < 8:
            toAppend_list.append([0,0,0,0,0,0,0,0])

        # Insert Month and Day of week for each Day
        month = [data_df[data_df['date'] == date]['month'].values[0]] + [0] * 7
        dayOfWeek = [data_df[data_df['date'] == date]['dayOfWeek'].values[0]] + [0] * 7
        toAppend_list.append(month)
        toAppend_list.append(dayOfWeek)
                
        dataX.append(toAppend_list)

    return np.array(dataX)

def main():
    #for num_split in [1,2,3,4,5]:
        #combineTVT(num_split)
    
    num_split = 1

    # Load data and drop the time column
    X_train = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/training-input.csv")
    X_test = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/test-input.csv")
    X_valid = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/valid-input.csv")
    y_train = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/training-output.csv").drop(columns=['date'])
    y_test = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/test-output.csv").drop(columns=['date'])
    y_valid = pd.read_csv(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/valid-output.csv").drop(columns=['date'])
    
    # Transform the values in ['date'] from string object to datetime object
    X_train['date'] = pd.to_datetime(X_train['date'])
    X_test['date'] = pd.to_datetime(X_test['date'])
    X_valid['date'] = pd.to_datetime(X_valid['date'])


    # Input data has two vector
    # 1: Vector of obervations over past 7 days
    X_train_Xt = looking_back(X_train[["date","TEMP","VISIB","WDSP","RH","wdir","pres","population_density","number_of_vehicles","month","dayOfWeek"]], 7)
    X_test_Xt = looking_back(X_test[["date","TEMP","VISIB","WDSP","RH","wdir","pres","month","dayOfWeek"]], 7)
    X_valid_Xt = looking_back(X_valid[["date","TEMP","VISIB","WDSP","RH","wdir","pres","month","dayOfWeek"]], 7)

    # 2: Binary Vector of regulations statuses
    X_train_I = X_train.drop(["date","TEMP","VISIB","WDSP","RH","wdir","pres","year","month","dayOfWeek","population_density","number_of_vehicles"], axis=1)
    X_test_I = X_test.drop(["date","TEMP","VISIB","WDSP","RH","wdir","pres","year","month","dayOfWeek","population_density","number_of_vehicles"], axis=1)
    X_valid_I = X_valid.drop(["date","TEMP","VISIB","WDSP","RH","wdir","pres","year","month","dayOfWeek","population_density","number_of_vehicles"], axis=1)

    # Turn outputs from dataframe into arrays
    y_train = y_train.values
    y_test = y_test.values
    y_valid = y_valid.values

    # Check shape
    print("X_train_Xt shape: ", X_train_Xt.shape)
    print("X_train_I shape: ", X_train_I.shape)
    print("y_train shape: ", y_train.shape)
    
    trainData_list = [X_train_Xt, X_train_I, y_train]
    validData_list = [X_valid_Xt, X_valid_I, y_valid]
    testData_list = [X_test_Xt, X_test_I, y_test]
    #tune_hyper(trainData_list, validData_list)

    #compute_ate(trainData_list, validData_list, testData_list)
    #batch_size = 32
    model = build_model(X_train_Xt, X_train_I, plot=True)
    #trained_model = train_model(model, X_train_Xt, X_train_I, y_train, batch_size)
    #trained_model.save(f"~/Documents/FYP-causal/bayesian-self/IOdata{num_split}/model{num_split}.h5")
    #print(f"\nmodel{num_split}.h5 saved\n")
    #evaluate_model(trained_model, X_valid_Xt, X_valid_I, y_valid, num_split, batch_size)



if __name__ == "__main__":
    main()