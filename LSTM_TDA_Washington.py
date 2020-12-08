

#%% Libraries
import numpy as np 
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import  DateFormatter

# To measure execution time...
import time
start_time = time.time()

#%% Parameters (
### HOSPITALIZATION - WEEKLY)
#LBack = 2 # Look back >=1 
#LPred = 3 # Future prediction >=0 ... 0 means X(t) -> Y(t+1) 
#pathFolderSaved = 'Saved/' 
## Dataset and State (HOSPITALIZATION - WEEKLY)
#nameState = 'Washington'
#nameDataset = 'InputLSTM/Washington_Hospitalization_TDA.csv'
## Train and test day 
#dayTrain_ini = date(2020,4,19) # Train INI
#dayTrain_fin = date(2020,8,30) # Train FINAL
#dayTest_ini = date(2020,9,6) # Test INI
#dayTest_fin = date(2020,9,27) # Test FINAL
## Location Variables
#locMAIN = [0, 37] # Main Variables
#locTDA = [38, 52] # TDA Variables
#locEV = [53, 64] # Environmental Variables

### (OTHERS - DAILY)
LBack = 4 # Look back >=1 
LPred = 29 # Future prediction >=0 ... 0 means X(t) -> Y(t+1) 
pathFolderSaved = 'Saved/'
# Dataset and State 
nameState = 'Washington'
#nameDataset = 'InputLSTM/Washington_MR_TDA.csv'
#nameDataset = 'InputLSTM/Washington_IFR_TDA.csv'
nameDataset = 'InputLSTM/Washington_NewCases_TDA.csv'
# Train and test day 
dayTrain_ini = date(2020,4,15) # Train INI
dayTrain_fin = date(2020,8,31) # Train FINAL
dayTest_ini = date(2020,9,1) # Test INI
dayTest_fin = date(2020,9,30) # Test FINAL
# Location Variables
locMAIN = [0, 40] # Main Variables
locTDA = [41, 55] # TDA Variables
locEV = [56, 67] # Environmental Variables

#%% Open the dataset
dfWhole = pd.read_csv(nameDataset) 
# --- Response variable 
# Select columns 
#df_Y = dfWhole # All variables...
# *** We have next options ***
df_Y = dfWhole[dfWhole.columns[list(range(locMAIN[0],locMAIN[1]+1))]] # Only Main variables...
# Date as Index -> 0
df_Y.Day = pd.to_datetime(df_Y.Day, format='%d/%m/%Y')
df_Y = df_Y.set_index("Day")
# --- Independent variable 
# Select columns
#selVariables = list(range(locMAIN[0],locMAIN[1]+1)) # Main Variables
#selVariables = list(range(locMAIN[0],locMAIN[1]+1)) + list(range(locEV[0],locEV[1]+1)) # Main + EV
selVariables = list(range(locMAIN[0],locMAIN[1]+1)) + list(range(locTDA[0],locTDA[1]+1)) # Main + TDA

#selVariables = list(range(locMAIN[0],locMAIN[1]+1)) + list(range(locEV[0],locEV[1]+1)) + list(range(locTDA[0],locTDA[1]+1)) # Main + EV + TDA

#df_X = dfWhole # All variables...
df_X = dfWhole[dfWhole.columns[selVariables]] # Only selected variables

# Date as index
df_X.Day = pd.to_datetime(df_X.Day, format='%d/%m/%Y') 
df_X = df_X.set_index("Day") 

# --- Only keep the training set
df_Y = df_Y[df_Y.index>=np.datetime64(dayTrain_ini)]
df_Y = df_Y[df_Y.index<=np.datetime64(dayTrain_fin)]
df_X = df_X[df_X.index>=np.datetime64(dayTrain_ini)]
df_X = df_X[df_X.index<=np.datetime64(dayTrain_fin)]

# --- Some variables
NTotal = len(df_Y) 
NFeatures = df_X.shape[1] 
NFeaturesOutput = df_Y.shape[1]

#%% Plot of time series 
ax = df_X.plot(kind='bar', figsize=(80,8), legend=True) 
plt.title('Variables ('+nameState+')') 
plt.ylabel('Values') 
plt.xlabel('Days') 
ticklabels = [item.strftime('%b %d') for item in df_X.index]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.xticks(fontsize=9, rotation=90)
plt.tight_layout()
#plt.show()
plt.savefig(pathFolderSaved+'Variables.pdf', bbox_inches='tight')
plt.close()

#%%
#plt.plot(df_Y)
ax = df_Y.plot(figsize=(40,6), legend=True) 
plt.title(nameState)
plt.xlabel('Days')
plt.ylabel('# Cases')
plt.xticks(df_Y.index.values)
ticklabels = [item.strftime('%b %d') for item in df_X.index]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
plt.xticks(fontsize=9, rotation=90)
plt.tight_layout()
#plt.show() 
plt.savefig(pathFolderSaved+'Main.pdf', bbox_inches='tight')
plt.close()

#%% Only Train 
# Unnormalized data
dataNorm_X = df_X.values 
dataNorm_Y = df_Y.values 
# Split into train and test sets 
dataX, dataY = [], [] 
dataX = np.zeros((NTotal-LBack+1, LBack*NFeatures))
for i in range(NTotal-LBack+1):
    dataX[i,:] = dataNorm_X[i:(i+LBack),:].reshape((1,LBack*NFeatures))

dataY = np.zeros((NTotal-LBack-LPred, NFeaturesOutput)) 
for i in range(NTotal-LBack-LPred):
    dataY[i,:] = dataNorm_Y[i+LBack+LPred,:]

train_size = dataY.shape[0]
pred_size =  dataX.shape[0] - train_size
trainX = dataX[0:train_size,:]
trainY = dataY[0:train_size,:]
predX = dataX[train_size:(train_size+pred_size),:]

# %% Prepare for LSTM
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
predX = np.reshape(predX, (predX.shape[0], 1, predX.shape[1]))

#%% Create and fit the LSTM network
model = Sequential()
batch_size = 8 
pDrop = 0.2
#valPatience = 400 # Hospitalization
valPatience = 800 # Others
#valPatience = 3000

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=valPatience) 
mc = ModelCheckpoint(pathFolderSaved+'best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True) 

#model.add(LSTM(256, activation='linear', input_shape=(1, LBack*NFeatures))) # OLD

#model.add(LSTM(512, activation='linear', return_sequences=True, input_shape=(1, LBack*NFeatures), dropout=pDrop))
#model.add(LSTM(256, activation='linear', return_sequences=True, dropout=pDrop)) 
#model.add(LSTM(256, activation='linear', dropout=pDrop))

# Hospitalization
#model.add(LSTM(64, activation='linear', return_sequences=True, input_shape=(1, LBack*NFeatures), dropout=pDrop))
#model.add(LSTM(32, activation='linear', return_sequences=True, dropout=pDrop)) 
#model.add(LSTM(32, activation='linear', dropout=pDrop))

# Others
model.add(LSTM(256, activation='linear', return_sequences=True, input_shape=(1, LBack*NFeatures), dropout=pDrop))
model.add(LSTM(128, activation='linear', return_sequences=True, dropout=pDrop)) 
model.add(LSTM(128, activation='linear', dropout=pDrop))

#model.add(Dense(1)) # A solely output...
model.add(Dense(trainY.shape[1])) # Multiple Outputs...

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error']) 
# Hospitalization
#history = model.fit(trainX, trainY, epochs=6000, batch_size=batch_size, verbose=0, callbacks=[es,mc], validation_data=(trainX, trainY))  
# Others
history = model.fit(trainX, trainY, epochs=8000, batch_size=batch_size, verbose=0, callbacks=[es,mc], validation_data=(trainX, trainY))  

#history = model.fit(trainX, trainY, epochs=10000, batch_size=batch_size, verbose=0, callbacks=[es,mc], validation_split=0.1)  

#%% plot history
plt.plot(history.history['loss'], label='train') 
plt.plot(history.history['val_loss'], label='validation')
plt.title(nameState)
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.legend()
#plt.show()
plt.savefig(pathFolderSaved+'History.pdf', bbox_inches='tight')
plt.close()

#%% To test if we saved the best model
saved_model = load_model(pathFolderSaved+'best_model.h5')
# trainPred = saved_model.predict(trainX)
# plt.plot(trainPredict)
# plt.plot(trainPred)
# plt.show()

# %% Predictions
# make predictions 
trainPredict = saved_model.predict(trainX) 
predPredict = saved_model.predict(predX) 
#****** Make train predict >=0 ****** IMPORTANT
#trainPredict[trainPredict<0]=0
# invert predictions 
trainPredict = trainPredict
trainY = trainY
predPredict = predPredict
# Mean squared error  
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore)) 

#%% Open real future 
#dfWholeTS = pd.read_csv('Datasets/TS_NC_TDA_CasesCountyJuly31.csv') 
dfWholeTS = pd.read_csv(nameDataset) 
dfFixTS = pd.DataFrame(dfWholeTS) 
# Date as index 
dfFixTS.Day = pd.to_datetime(dfFixTS.Day, format='%d/%m/%Y') 
dfFixTS = dfFixTS.set_index("Day") 
#dfFixTS = dfFixTS[['Alamance']].astype(float) 
dfFixTS = dfFixTS.astype(float) 
# Table                
dfTS = dfFixTS 
TSfuture = dfTS.values 
TSfut_size = len(TSfuture) 
# Plot of time series
#plt.plot(dfTS)
#plt.show()

#%% Datasets in Pandas (Save predictions)
# --- CSV files 
df_trainPredict = pd.DataFrame(trainPredict, columns=df_Y.columns, index=dfFixTS.index[LBack+LPred:LBack+LPred+train_size])
df_predPredict = pd.DataFrame(predPredict, columns=df_Y.columns, index=dfFixTS.index[LBack+LPred+train_size:LBack+LPred+train_size+pred_size]) 
#df_trainPredict.to_csv(pathFolderSaved+'NCAROLINA_trainPredict.csv') 
#df_predPredict.to_csv(pathFolderSaved+'NCAROLINA_predPredict.csv') 
df_trainPredict.to_csv(pathFolderSaved+nameState+'_trainPredict.csv')
df_predPredict.to_csv(pathFolderSaved+nameState+'_predPredict.csv') 
# --- PLOT 
df_Y.plot(figsize=(40,6), legend=True)
df_trainPredict.plot(figsize=(40,6), legend=True)  
df_predPredict.plot(figsize=(40,6), legend=True)

#%%

#%% Timing
print("\n TIME: "+str((time.time() - start_time))+" Seg ---  "+str((time.time() - start_time)/60)+" Min ---  "+str((time.time() - start_time)/(60*60))+" Hr ")

# %% 
