import os
import pickle
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import time
import scipy.io
import re
from termcolor import colored 

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

def get_weather_data():
    ## load all hourly climatic data expect pressure data which is incomplete
    weather_data = read_csv('Pittsburgh_2stations_climate_18-19.csv',header=0,usecols=[1,41,42,43,44,45,48,49,50,51,52,53,54,55,56],nrows=27315) #27315
    
    weather_data = weather_data.dropna(how='all')
    ####### Date and time
    date = weather_data['DATE'].astype(str).str.replace('T', '-')
    date = date.astype(str).str.split('-',expand=True)
    date = date.rename(columns={0:'Year',1:'Month',2:'Day',3:'Time'})

    time = date['Time'].astype(str).str.split(':',expand=True)
    time = time.rename(columns={0:'Hour',1:'Minute',2:'Second'})

    date = date.drop('Time',axis=1)
    time = time.drop('Second',axis=1)
    weather_data = weather_data.drop('DATE',axis=1)
    weather_data = pd.concat([date,time,weather_data],axis=1)

    weather_data['HourlyAltimeterSetting'].update(weather_data['HourlyAltimeterSetting'].astype(float).interpolate())
    weather_data['HourlyDewPointTemperature'].update(weather_data['HourlyDewPointTemperature'].astype(float).interpolate().astype(int))
    weather_data['HourlyDryBulbTemperature'].update(weather_data['HourlyDryBulbTemperature'].astype(float).interpolate().astype(int))
    weather_data['HourlyRelativeHumidity'].update(weather_data['HourlyRelativeHumidity'].astype(float).interpolate().astype(int))
    weather_data['HourlySeaLevelPressure'].update(weather_data['HourlySeaLevelPressure'].astype(float).interpolate())
    weather_data['HourlyStationPressure'].update(weather_data['HourlyStationPressure'].astype(float).interpolate())
    weather_data['HourlyVisibility'].update(weather_data['HourlyVisibility'].astype(float).interpolate())
    weather_data['HourlyWetBulbTemperature'].update(weather_data['HourlyWetBulbTemperature'].astype(float).interpolate().astype(int))
    weather_data['HourlyWindGustSpeed'].fillna(0, inplace=True)
    weather_data['HourlyWindSpeed'].update(weather_data['HourlyWindSpeed'].astype(float).interpolate().astype(int))
        
    ####### Precipitation replace T by 0.001
    weather_data['HourlyPrecipitation'].update(weather_data['HourlyPrecipitation'].astype(str).str.replace('T','0.001'))
    weather_data['HourlyPrecipitation'].update(weather_data['HourlyPrecipitation'].astype(str).str.replace('s',''))
    weather_data['HourlyPrecipitation'].update(weather_data['HourlyPrecipitation'].astype(float).interpolate())

    ####### Process weather type, keep only AU code without description +-VC ######
    temp = weather_data['HourlyPresentWeatherType'].astype(str).str.split('|',expand=True)

    temp[0] = temp[0].mask(temp[0].str.len() == 0, temp[1])
    temp[0] = temp[0].mask(temp[0].str.len() == 0, temp[2])
    temp = temp[0].astype(str).str.replace(':', '')
    temp = temp.astype(str).str.replace('+', '')
    temp = temp.astype(str).str.replace('-', '')
    temp = temp.astype(str).str.replace('VC', '')
    temp = temp.astype(str).str.replace('\d+', '')
    temp = temp.str.split(' ')
    
    temp = pd.get_dummies(temp.apply(pd.Series).stack()).sum(level=0) # all type ['SN','RA','DZ','SG','IC','PL','GR','GS','UP','BR','FG','FU','VA','DU','SA','HZ','PY','PO','SQ','FC','SS','DS']
    temp = temp.drop('nan',axis=1)
    temp = temp.drop('',axis=1)
    weather_type = temp.head(1)
    weather_data = pd.concat([weather_data,temp],axis=1)
    weather_data = weather_data.drop('HourlyPresentWeatherType',axis=1)

    ####### Process sky condition ####### take the third layer as full state ignore cloud height
    temp = weather_data['HourlySkyConditions'].astype(str).str.replace('\d+', '')
    temp = temp.astype(str).str.replace(' ', '')
    temp = temp.astype(str).str.split(':',expand=True)
    temp[2].fillna(value='', inplace=True)
    temp[2] = temp[2].mask(temp[2].str.len() == 0, temp[1])
    temp[2] = temp[2].mask(temp[2].str.len() == 0, temp[0])
    temp = pd.get_dummies(temp[2])

    sky_condition = temp.head(1)
    weather_data = pd.concat([weather_data,temp],axis=1)
    weather_data = weather_data.drop('HourlySkyConditions',axis=1)

    ######## Wind direction replace VRB by 370
    weather_data['HourlyWindDirection'].update(weather_data['HourlyWindDirection'].ffill())
    weather_data['HourlyWindDirection'].update(weather_data['HourlyWindDirection'].astype(str).str.replace('VRB','370').astype(int))
    weather_value = weather_data.to_numpy()


    ######## Keep only weather data collected from hour:51:00
    #weather_data = weather_data[weather_data['Minute'].astype(int)==51]
    weather_data['Hour'] = np.where((weather_data['Minute'].astype(int)!=0),weather_data['Hour'].astype(int)+1,weather_data['Hour'])
    weather_data.loc[weather_data['Hour'].astype(int)==24,'Hour'] = 0
    weather_data.loc[weather_data['Minute'].astype(int)!=0,'Minute'] = 0

    weather_data = weather_data.astype('float').groupby(['Year','Month','Day','Hour']).mean().reset_index()

    return weather_data,weather_type,sky_condition

def get_load_data():
    load2018 = read_csv('Load_Duq_2018.csv',header=0,usecols=[1,6])
    load2018 = load2018.rename(columns={'datetime_beginning_ept':'DATE','mw':'Load'})
    load2019 = read_csv('Load_Duq_2019.csv',header=0,usecols=[1,6])
    load2019 = load2019.rename(columns={'datetime_beginning_ept':'DATE','mw':'Load'})

    load_data = pd.concat([load2018,load2019],axis=0)
    time = load_data['DATE'].astype(str).str.split(' ',expand=True)
    date = time[0].astype(str).str.split('/',expand=True)
    date = date.rename(columns={0:'Month',1:'Day',2:'Year'})

    time_temp = time[1].astype(str).str.split(':',expand=True)
    time_temp = time_temp.rename(columns={0:'Hour',1:'Minute',2:'Second'})

    #time_temp['Hour'] = np.where((time_temp['Hour'].astype(int)==12) & (time[2].astype(str)=='AM'),0,time_temp['Hour'])
    #time_temp['Hour'] = np.where((time_temp['Hour'].astype(int)!=12) & (time[2].astype(str)=='PM'),time_temp['Hour'].astype(int)+12,time_temp['Hour'])
    #time_temp = time_temp.drop('Second',axis=1)

    load_data = load_data.drop('DATE',axis=1)
    load_data = pd.concat([date,time_temp,load_data],axis=1)

    ### Change DST time back to EST time (standard time) to match with weather
    return load_data

def match_weather_load(weather_data,load_data,all_data_file):
    

    load_data['Year'] = load_data['Year'].astype(int)
    load_data['Month'] = load_data['Month'].astype(int)
    load_data['Day'] = load_data['Day'].astype(int)
    load_data['Hour'] = load_data['Hour'].astype(int)
    load_data['Minute'] = load_data['Minute'].astype(int)

    weather_data['Year'] = weather_data['Year'].astype(int)
    weather_data['Month'] = weather_data['Month'].astype(int)
    weather_data['Day'] = weather_data['Day'].astype(int)
    weather_data['Hour'] = weather_data['Hour'].astype(int)
    weather_data['Minute'] = weather_data['Minute'].astype(int)

    all_data = pd.merge(weather_data,load_data,'outer',on=['Year','Month','Day','Hour','Minute'])
    all_data.to_csv(all_data_file,index=False)


class Encoder(Model):
    def __init__(self):
        super(Encoder,self).__init__()
        self.dense0 = Dense(16,use_bias=True,activation='relu')
        self.dense1 = Dense(10,use_bias=True,activation='relu')
        self.dense2 = Dense(6,use_bias=True,activation='sigmoid')

    def call(self, x):
        y = self.dense0(x)
        y = self.dense1(y)
        y = self.dense2(y)
        return y

class Decoder(Model):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dense0 = Dense(10,use_bias=True,activation='relu')
        self.dense1 = Dense(16,use_bias=True,activation='relu')
        self.dense2 = Dense(input_shape,use_bias=True,activation='sigmoid')

    def call(self, y):
        x = self.dense0(y)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

@tf.function
def train_AE_step(inputs):
    loss = 0
    # inputs shape [batch_size,input_shape]
    with tf.GradientTape() as tape:

        encoded = Encoder_model(inputs)
        output = Decoder_model(encoded)
        loss = 0.9*loss_object(output[:,:12],inputs[:,:12])+0.1*loss_object(output[:,12:],inputs[:,12:])

    variables = Encoder_model.trainable_variables + Decoder_model.trainable_variables

    gradients = tape.gradient(loss,variables)
    grad_norm = tf.linalg.global_norm(gradients)

    optimizer.apply_gradients(zip(gradients,variables))

    output = output*(weather_max-weather_min) + weather_min
    mse = tf.reduce_mean(tf.square(inputs*(weather_max-weather_min)+ weather_min-output))
    #mse = tf.reduce_mean(tf.square(inputs-output))

    return mse,loss,grad_norm

def evaluate_AE(target,target_complete):
    # Reverse normalization
    encoded = Encoder_model(target)
    output = Decoder_model(encoded)
    output = output.numpy()*(weather_max-weather_min) + weather_min

    # set threshold for last 13+6 weather attributes
    thre = 0.7
    
    write_message = ''

    for i in range(np.shape(output)[0]):
        wtype_index = np.where(output[i,12:25] > thre)
        stype_index = np.where(output[i,25:] > thre)

        wtype_target_index = np.where(target_complete[i,12:25] > thre)
        stype_target_index = np.where(target_complete[i,25:] > thre)

        write_message += '%dth weather truth:      '%i
        for j in range(12):
            write_message += '%.3f '%(target_complete[i,j])
        if wtype_target_index[0].size==0:
            write_message += ' No weather type '
        else:
            write_message += ' '.join([weather_type[j] for j in wtype_target_index[0]])+' '
        if stype_target_index[0].size==0:
            write_message += ' No sky condition '
        else:
            write_message += ' '.join([sky_condition[j] for j in stype_target_index[0]])+' '

        write_message += '\n' 
        write_message += '%dth weather prediction: '%i
        for j in range(12):
            write_message += '%.3f '%(output[i,j])
        if wtype_index[0].size==0:
            write_message += ' No weather type '
        else:
            write_message += ' '.join([weather_type[j] for j in wtype_index[0]])+' '
        if stype_index[0].size==0:
            write_message += ' No sky condition '
        else:
            write_message += ' '.join([sky_condition[j] for j in stype_index[0]])+' '
        write_message += '\n\n' 

    with open(os.path.join(result_path,'Evaluation_epoch%d.txt'%epoch),'w') as filehandle:
        filehandle.write(write_message)



all_data_file = 'Load_and_weather_data_1819.csv'
#weather_data, weather_type, sky_condition = get_weather_data()
if not os.path.exists(all_data_file):
    load_data = get_load_data()
    weather_data, weather_type, sky_condition = get_weather_data()
    match_weather_load(weather_data,load_data,all_data_file)

weather_type = ['BC','BL','BR','DZ','FG','FZ','HZ','MI','PL','RA','SN','SQ','TS']
sky_condition = ['BKN','CLR','FEW','OVC','SCT','VV']

weather_data = read_csv(all_data_file,header=0,usecols=np.arange(5,36))
weather_data = weather_data.to_numpy()
load_data = read_csv(all_data_file,header=0,usecols=[36])
load_data = load_data.to_numpy()

weather_data = np.array(weather_data).astype('float32')
data_size = np.shape(weather_data)[0] # weather data of size 31, 12+13+6, 13 weather types maximum 4 type minimal 0, 6 sky condition max 1 condition min 0
input_shape = np.shape(weather_data)[1]

# Normalization of weather data to [0,1]
weather_min = np.min(weather_data,axis=0)
weather_max = np.max(weather_data,axis=0)

weather_norm = (weather_data-weather_min)/(weather_max-weather_min)

# Normalization of load data to [0,1]
load_min = np.min(load_data,axis=0)
load_max = np.max(load_data,axis=0)

load_norm = (load_data-load_min)/(load_max-load_min)

'''
Train Autoencoder
'''

nb_feature = 8
learning_rate = 0.002
batch_size = 50


data_size = np.shape(weather_data)[0] # weather data of size 31, 12+13+6, 13 weather types maximum 4 type minimal 0, 6 sky condition max 1 condition min 0
input_shape = np.shape(weather_data)[1]

result_path = '3MLP10_Adam_lr_%.5f_batch%d_weightloss82_sigmoid'%(learning_rate,batch_size)
if not os.path.exists(result_path):
    os.mkdir(result_path)

Training_log_file = 'training_log.txt'

with open(os.path.join(result_path,Training_log_file),'w') as filehandle:
    filehandle.write('Encoder of 3 layers MLP input_shape->16->10->6\n')
    filehandle.write('activation relu relu sigmoid\n')
    filehandle.write('Discriminator of 3 layers MLP 10->16->input_shape\n')
    filehandle.write('activation relu relu sigmoid \n')
    filehandle.write('Add weight to loss correspond to continuous data or distrete (one-hot) \n')
    filehandle.write('Learning rate: '+str(learning_rate)+'\n'+'batch size: '+str(batch_size)+'\n')
    
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_object = tf.keras.losses.MeanSquaredError()

Dataset = tf.data.Dataset.from_tensor_slices((weather_norm))
Dataset = Dataset.shuffle(10000)
Dataset = Dataset.batch(batch_size)
steps_per_epoch = data_size//batch_size

Encoder_model = Encoder()
Decoder_model = Decoder()
losses = []

for epoch in range(1,300+1):
    start = time.time()
        
    total_loss = 0
    total_mse = 0
    for (batch, (targ)) in enumerate(Dataset.take(steps_per_epoch)):
        batch_mse, batch_loss,grad_norm = train_AE_step(targ)
        total_loss += batch_loss
        total_mse += batch_mse            

    losses.append(total_loss / steps_per_epoch)
    if epoch % 1 == 0:
        print('Gradient norm'+str(grad_norm))
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
        print('Epoch {} MSE {:.4f}'.format(epoch + 1,total_mse / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    if epoch % 50 == 0:

        encoder_wfile = 'ENweights_epoch_'+str(epoch)+'.hdf5'
        decoder_wfile = 'DEweights_epoch_'+str(epoch)+'.hdf5'
        Encoder_model.save_weights(os.path.join(result_path,encoder_wfile), True)
        Decoder_model.save_weights(os.path.join(result_path,decoder_wfile), True)

        plt.plot(losses,label="mse loss")
        plt.legend()
        plot_name = 'training_loss_epoch'+str(epoch)+'.png'
        plt.savefig(os.path.join(result_path,plot_name))
        plt.close()

        scipy.io.savemat(
            os.path.join(result_path,'Training history.mat'),
            {'loss':np.array(losses)}
        )

        indices = np.arange(data_size)
        np.random.shuffle(indices)
        evaluate_AE(weather_norm[indices[:5]],weather_data[indices[:5]])


