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
from tensorflow.keras.layers import Dense, LSTM, GRU


class build_rnn(Model):
    def __init__(self):
        super(build_rnn,self).__init__()
        self.rnnLL = LSTM(hidden_units,return_state=True)
        self.rnnWL = LSTM(hidden_units,return_sequences=True)
        #self.rnn1 = LSTM(hidden_units,return_sequences=True)
        #self.dense1 = Dense(20,use_bias=True,activation='sigmoid')
        self.dense2 = Dense(1,use_bias=True)
    

    def call(self, x):
        '''
        input of shape ntime*(nfeat=6)
        output of shape ntime*1 -> prediction of load 
        '''
        _,h,c = self.rnnLL(x)
        w = tf.zeros([tf.shape(x)[0],ntime,1])
        y = self.rnnWL(w,initial_state=(h,c))
        #y = self.rnn1(y)
        #y = self.dense1(y)
        y = self.dense2(y)
        return y 


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

@tf.function
def train_rnn_step(input_load,targets):
    loss = 0
    # inputs shape [batch_size,input_shape]
    with tf.GradientTape() as tape:
        outputs = rnn_model(input_load)
        loss = loss_object(outputs,targets)

    variables = rnn_model.trainable_variables

    gradients = tape.gradient(loss,variables)
    grad_norm = tf.linalg.global_norm(gradients)

    optimizer.apply_gradients(zip(gradients,variables))

    outputs = outputs*(load_max-load_min) + load_min
    mse = tf.reduce_mean(tf.square(targets*(load_max-load_min)+ load_min-outputs))
    #mse = tf.reduce_mean(tf.square(inputs-output))

    return mse,loss,grad_norm


def evaluate_rnn(input_load,targets,training):
    # Reverse normalization
    outputs = rnn_model(input_load)
    outputs = outputs*(load_max-load_min)+ load_min
    targets = targets*(load_max-load_min)+ load_min

    RMSE = tf.sqrt(tf.reduce_mean(tf.square(targets-outputs)))
    MAPE = tf.reduce_mean(tf.abs(targets-outputs)/tf.abs(targets))
    MAE = tf.reduce_mean(tf.abs(targets-outputs))

    if epoch%50 == 0:
        for i in range(5):
            plt.subplot(5,1,i+1)
            Real_load, = plt.plot(outputs[i,:].numpy(),'m',label='Prediction') 
            Pred_load, = plt.plot(targets[i,:],'c',label='True load')
            plt.title('%.3f'%(np.mean(np.square(outputs[i,:].numpy()-targets[i,:].numpy()))) )
        plt.legend(handles=[Real_load,Pred_load])
        
        if training:
            plot_name = 'prediction_train_'+str(epoch)+'.png'
        else:
            plot_name = 'prediction_test_'+str(epoch)+'.png'

        plt.savefig(os.path.join(result_path,plot_name))
        plt.close()

    return RMSE, MAPE, MAE

''' 
Train RNN for prediction
'''

all_data_file = 'Load_and_weather_data_1819.csv'

weather_data = read_csv(all_data_file,header=0,usecols=np.arange(5,36))
weather_data = weather_data.to_numpy()
load_data = read_csv(all_data_file,header=0,usecols=[36])
load_data = load_data.to_numpy()
load_data = np.array(load_data).astype('float32')

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

## Get encoded weather features
AE_result_path = '3MLP10_Adam_lr_0.00200_batch50_weightloss82_sigmoid'
feature_file = 'Encoded_weather.pkl'
if not os.path.exists(os.path.join(AE_result_path,feature_file)):
    Encoder_model = Encoder()
    Encoder_model.build(tf.TensorShape([50,input_shape]))
    weather_features = Encoder_model(weather_norm)
    output = open(os.path.join(AE_result_path,feature_file),'wb')
    pickle.dump(weather_features.numpy(),output)
    output.close()
else:
    weather_features = pickle.load(open(os.path.join(AE_result_path,feature_file),'rb'))

### Prepare for trianing RNN
hidden_units = 80
learning_rate = 0.002
batch_size = 60

ntime = 24 # prediction for one week
nprev = 24*3 # use 4 week previous load to predict current week
ndata = (data_size-nprev-ntime)//24 +1

weather_feat_curr = np.zeros([ndata,ntime,6],dtype=np.float32)
load_norm_curr = np.zeros([ndata,ntime,1],dtype=np.float32)
load_norm_prev = np.zeros([ndata,nprev,1],dtype=np.float32)

for i in range(ndata):
    weather_feat_curr[i,:,:] = weather_features[i*24+nprev:i*24+nprev+ntime,:]
    load_norm_curr[i,:,:] = load_norm[i*24+nprev:i*24+nprev+ntime,:]
    load_norm_prev [i,:,:] = load_norm[i*24:i*24+nprev,:]

test_size = 60 #take last 2 months data for testing
train_size = ndata - test_size

train_dataset = tf.data.Dataset.from_tensor_slices((load_norm_prev[:train_size,:,:],load_norm_curr[:train_size,:,:]))
test_dataset = tf.data.Dataset.from_tensor_slices((load_norm_prev[train_size:,:,:],load_norm_curr[train_size:,:,:]))
train_dataset = train_dataset.shuffle(10000)

train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(test_size)
steps_per_epoch = train_size//batch_size


result_path = 'WL_prevStateNew_withoutweat_1LSTM%d_MLP_%dntime%d_Adam_lr%.5f'%(hidden_units,nprev,ntime,learning_rate)
if not os.path.exists(result_path):
    os.mkdir(result_path)

Training_log_file = 'training_log.txt'

with open(os.path.join(result_path,Training_log_file),'w') as filehandle:
    filehandle.write('2 layer LSTM\n')
    filehandle.write('1 layer MLP -> 1\n')
    filehandle.write('The data was taken 24h step\n')
    filehandle.write('Feed prev load to rnn1 to get initila states\n')
    filehandle.write('activation -> none\n')
    filehandle.write('previous length of 24*7*4 4weeks\n')
    filehandle.write('predict length of 24*7 1weeks\n')
    filehandle.write('Learning rate: '+str(learning_rate)+'\n'+'batch size: '+str(batch_size)+'\n')
    
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_object = tf.keras.losses.MeanSquaredError()

rnn_model = build_rnn()
losses = []
train_mse = []

test_rmse = []
test_mape = []
test_mae = []

for epoch in range(1,600+1):
    start = time.time()
        
    total_loss = 0
    total_mse = 0
    for (batch, (inp_load,targ)) in enumerate(train_dataset.take(steps_per_epoch)):
        batch_mse, batch_loss,grad_norm = train_rnn_step(inp_load,targ)
        total_loss += batch_loss
        total_mse += batch_mse

    losses.append(total_loss / steps_per_epoch)
    train_mse.append(total_mse / steps_per_epoch)

    ep_rmse, ep_mape, ep_mae = evaluate_rnn(inp_load,targ,training=True)            


    for (batch, (inp_load,targ)) in enumerate(test_dataset.take(1)):
        curr_test_rmse, curr_test_mape, curr_test_mae = evaluate_rnn(inp_load,targ,training=False)
    test_rmse.append(curr_test_rmse.numpy())
    test_mape.append(curr_test_mape.numpy())
    test_mae.append(curr_test_mae.numpy())

    

    if epoch % 1 == 0:
        print('Gradient norm'+str(grad_norm))
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
        print('Epoch {} MSE {:.4f}'.format(epoch + 1,total_mse / steps_per_epoch))
        print('Epoch {} RMSE {:.4f} MAPE {:.4f} MAE {:.4f}'.format(epoch + 1,ep_rmse.numpy(),ep_mape.numpy(),ep_mae))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    if epoch % 50 == 0:

        rnn_wfile = 'RNNweights_epoch_'+str(epoch)+'.hdf5'
        rnn_model.save_weights(os.path.join(result_path,rnn_wfile), True)

        plt.plot(losses,label="mse loss")
        plt.legend()
        plot_name = 'training_loss_epoch'+str(epoch)+'.png'
        plt.savefig(os.path.join(result_path,plot_name))
        plt.close()

        scipy.io.savemat(
            os.path.join(result_path,'Training history.mat'),
            {'loss':np.array(losses),
            'test_rmse':np.array(test_rmse),
            'test_mape':np.array(test_mape),
            'test_mae':np.array(test_mae)}
        )

'''result_path = "WL_prevStateNew_withoutweat_1LSTM80_MLP_ntime72_Adam_lr0.00200"
epoch=600
rnn_model = build_rnn()
rnn_model.build(tf.TensorShape([50,ntime,1]))
wfile = 'RNNweights_epoch_'+str(epoch)+'.hdf5'    
rnn_model.load_weights(os.path.join(result_path,wfile))'''

inputs = load_norm_prev[train_size:,:,:]
targets = load_norm_curr[train_size:,:,:]*(load_max-load_min)+ load_min

outputs = rnn_model(inputs)*(load_max-load_min)+ load_min

for i in range(5):
    plt.subplot(5,1,i+1)
    index = i*10
    Real_load, = plt.plot(outputs[index,:].numpy(),'m',label='Prediction') 
    Pred_load, = plt.plot(targets[index,:],'c',label='True load')
    plt.title('%.3f'%(np.sqrt(np.mean(np.square(outputs[index,:].numpy()-targets[index,:])))) )
    plt.grid()
plt.legend(handles=[Real_load,Pred_load])
        
plot_name = 'prediction_Example.png'


plt.savefig(os.path.join(result_path,plot_name))
plt.close()