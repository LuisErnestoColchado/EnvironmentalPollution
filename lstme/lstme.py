#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:15:01 2019

@author: luisernesto
"""

#%%
import numpy as np 
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
#%%
df = pd.read_csv("data/BeijingPM20100101_20151231.csv")
#%%
data = df.iloc[:,[1,2,3,4,9,11,13,15]].values
#%%
#GET METEOROLOGICAL DATA 
meteorological = data[:,[6,7,5]]
#%%
scaler = MinMaxScaler(feature_range=(0, 1))
scaledMeteorological = scaler.fit_transform(meteorological)
#%%
#GET CATEGORICAL DATA
categorical_month = data[:,1].astype(int)
categorical_hour = data[:,3].astype(int)
#%%
#ENCODE CATEGORICAL DATA TO ONE-HOT
months = 12
hours = 24
onehot_month = np.zeros((len(data), months))
onehot_hours = np.zeros((len(data),hours))
onehot_month[np.arange(len(data)),categorical_month-1] = 1.0
onehot_hours[np.arange(len(data)),categorical_hour] = 1.0
#%%
auxData = np.concatenate((scaledMeteorological,onehot_month,onehot_hours),axis=1)
#%%
#PM2.5
pm25 = data[:,4]
pm25 = np.nan_to_num(pm25)
pm25 = np.reshape(pm25,(len(data),1))
#%%
scaler = MinMaxScaler(feature_range=(0, 1))
scaledPm25 = scaler.fit_transform(pm25)
#%%
DIV = len(data)/8
historicalData =  np.zeros((len(data),8))
historicalLabel = np.zeros((len(data),1))
row = 0
col = 0
for i in range (0,len(data)):
    j = 8
    col = 0
    while(i > 8 and j > 0):
        historicalData[row,col] = scaledPm25[i-j]
        col += 1
        j -= 1
    historicalLabel[row,0] = scaledPm25[i] 
    row += 1
#%%
#auxiliarData = np.zeros((int(len(auxData)/8),39))
#j = 0
#meanTemperature = 0
#meanHumidity = 0
#meanWS = 0 
#onehotValue = 0
#for i in range(0,int(len(auxData)/8)-1):
#    while(j < 8):
#        for k in range(0,len(auxData[0])):
#            if (k == 0):
#                meanTemperature += auxData[(i*8)+j,k]
#                auxiliarData[i,k] = meanTemperature/8  
#            elif(k == 1):
#                meanHumidity += auxData[(i*8)+j,k]
#                auxiliarData[i,k] = meanHumidity/8
#            elif(k == 2):
#                meanWS += auxData[(i*8)+j,k]
#                auxiliarData[i,k] = meanWS/8
#            else: 
#                onehotValue += auxData[(i*8)+j,k]
#                if(auxiliarData[i,k]  < onehotValue):
#                    auxiliarData[i,k] = onehotValue
#                onehotValue = 0 
#        meanTemperature = 0
#        meanHumidity = 0
#        meanWS = 0 
#
#        j+=1
#    j=0
#%%
xTrain = historicalData[0:40000,:]
yTrain = historicalLabel[0:40000,:]
xTest = historicalData[40000:52584,:]
yTest = historicalLabel[40000:52584,:]
auxTrain = auxData[0:40000,:]
auxTest = auxData[40000:52584,:]
#%%

#%%
learning_rate = np.power(10.0,-2.0)
training_steps = 10000
batch_size = 1000
display_step = 100
num_input = 8
timesteps = 1
num_units = 1000
num_classes = 1
num_layers = 2
#%%
tf.reset_default_graph()

X = tf.placeholder("float", [None, timesteps,num_input])
Y = tf.placeholder("float", [None,num_classes])
aux = tf.placeholder("float", [None,timesteps,39])

keep_prob = 1.0

weights = {
    'out': tf.Variable(tf.random_normal([num_units, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

initializer = tf.random_uniform_initializer(-1, 1)
def RNN(x, weights, biases):
    outputs = x
    #track through the layers
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer),reuse=tf.AUTO_REUSE):
            #forward cells
            cell_fw = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, activation=tf.nn.sigmoid,input_keep_prob = keep_prob)
            #backward cells
            cell_bw = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

            (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw, outputs,dtype=tf.float32)
            #lstm_cell = tf.contrib.rnn.LSTMCell(num_units,initializer=initializer)
            #lstm = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob = keep_prob)
            #outputs, states = tf.nn.static_rnn(lstm, x, dtype=tf.float32)
            #output = tf.matmul(outputs[-1], layers['weights']) + layers['bias']
    rnn_outputs_fw = tf.reshape(output_fw, [-1, num_units])
    rnn_outputs_bw = tf.reshape(output_bw, [-1, num_units])
    out_fw = tf.matmul(rnn_outputs_fw, weights['out']) + biases['out']
    out_bw = tf.matmul(rnn_outputs_bw, weights['out']) + biases['out']
    return np.add(out_fw,out_bw)    
    #output = np.asarray(np.add(out_fw,out_bw)).reshape((1,1))    
    #outLSTM = tf.matmul(output, weights['out']) + biases['out']    
    #rnn_outputs_fw = tf.reshape(output_fw, [-1, num_units])
    #rnn_outputs_bw = tf.reshape(output_bw, [-1, num_units])
    #out_fw = tf.matmul(rnn_outputs_fw, weights['out']) + biases['out']
    #out_bw = tf.matmul(rnn_outputs_bw, weights['out']) + biases['out']
    #outLSTM = np.add(out_fw,out_bw)
    #tf.layers.dense(outLSTM,)
logits = RNN(X, weights, biases)
prediction = tf.nn.sigmoid(logits)
shape = tf.shape(logits)
#! 
prediction = tf.reshape(logits,[shape[0],timesteps,1])
 
vector = tf.concat((prediction,aux),axis=2)
fc1 = tf.layers.dense(vector, units=1000, activation=tf.nn.sigmoid)
fc2 = tf.layers.dense(inputs=fc1, units=1, activation=tf.nn.sigmoid)
finalOut = tf.reshape(fc2,[batch_size,1])
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=finalOut, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#calculate gradients 
gvs = optimizer.compute_gradients(loss_op)

#clipping gradients 
def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)
capped = [(ClipIfNotNone(grad),var) for grad,var in gvs]
train_op = optimizer.apply_gradients(capped)
optimizer.minimize(loss_op)

p5 = tf.constant(0.5)
delta = tf.abs((Y - finalOut))
#accuracy = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(finalOut,Y),(Y + 1e-10))))
#accuracy = 1 - ecm
correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)

#calculate accuracy
#accuracy = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(prediction,Y),(Y + 1e-10))))
accuracy = tf.keras.backend.mean(tf.keras.metrics.mean_absolute_percentage_error(Y,prediction))
init = tf.global_variables_initializer()
#%%
accuracyTraining = []
lossTraining = []
steps = []
def startSession(xTrain,xTest,yTrain,yTest,auxTest,auxTrain):
    with tf.Session() as sess:
        sess.run(init)
        count = 0
        startData = 0
        endData = batch_size
        
        #test_len = 1073
        test_data = xTest[:,:].reshape((-1, timesteps, num_input))
        test_label = yTest[:].reshape((12584,1))
        test_aux = auxTest[:,:].reshape((-1, timesteps, 39))
        for step in range(1, training_steps+1):
            count+=1
            ##print("start ",startData)
            ##print("end ",endData)
            batch_x = xTrain[startData:endData,:]
            batch_aux = auxTrain[startData:endData,:]
            batch_y = yTrain[startData:endData]
            ##print(batch_x.shape)
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            batch_aux = batch_aux.reshape((batch_size, timesteps, 39))
            batch_y = batch_y.reshape((batch_size,1))
            batch_y[:,:].astype(float)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y,aux: batch_aux})
            
            if step % display_step == 0 or step == 1:
                loss, accu = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     aux: batch_aux})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training MAPE= " + \
                      "{:.3f}".format(accu))

                lossTraining.append(loss)
                steps.append(step)
                accu = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, aux: test_aux})
                print("Testing MAPE: ", accu)
                accuracyTraining.append(accuracy)
            if(endData<xTrain.shape[0]):
                startData=count*batch_size
                endData = (count+1)*batch_size
            else:
                count=0
                startData = 0
                endData = batch_size
        print("Optimization Finished!")
#%%

print("**********LTSM***********")
startSession()
#%%
### Cross validation 
kFolds = KFold(n_splits=5)
npTrainData = trainData.values
countCross = 1 
for train, test in kFolds.split(npTrainData):
    trainD = npTrainData[train,0:40]
    testD = npTrainData[test,0:40]
    yTrain = npTrainData[train,40:41]
    yTest = npTrainData[test,40:41]
    print("------------- CROSS VALIDATION ",countCross,"---------")
    print("**********LTSM***********")
    startSession()
#ltsm_loss_training = lossTraining
#ltsm_accuracy = accuracyTraining