seed_value=325
import os
os.environ['PYTHONHASHSEED' ] =str(seed_value)
#  Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

# Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, layers, callbacks,Model
from tensorflow.keras.layers import Layer,Dense, LSTM, Dropout,BatchNormalization,GRU, Bidirectional,SimpleRNN,Conv2D,Conv3D,Input,MaxPool3D,MaxPool2D,Flatten,TimeDistributed,Add,Lambda,Reshape,Softmax
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
import math
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
from set_sample_track_1 import shift,dis_direct,DenoisMat,minmax_scaler
import pandas as pd
import os, glob
from functools import reduce
import operator
import numpy as np
import random
import math
import datetime

import warnings
warnings.filterwarnings("ignore")

def getDistance(latA, lonA, latB, lonB):
    ra = 6378136.49  # radius of equator: meter
    rb = 6356755  # radius of polar: meter

    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)

    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    if radLonA == radLonB:
        x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(0.001))
    else:
        x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr) / 1000
    return distance


dataframe=pd.read_csv('./IBTrACS_droptime_usa_2010_2024.txt',sep=',',names=['name', 'date', 'lat', 'lon', 'ws', 'p', 'speed', 'direct'])
print(dataframe[0:20])
i_index=[]
for i in range(len(dataframe)):
    if dataframe['name'][i]=='66666':
        i_index.append(i)

for i in range(len(dataframe)):
    if dataframe['lon'][i]<0:
        dataframe['lon'][i]=dataframe['lon'][i]+360
#2479
j_index=[]
ii_index=[]
for i in range(len(i_index)):
    index=i_index[i]
    if i==len(i_index)-1:
        m=index+1
        n=len(dataframe)
    else:
        m=index+1
        n=i_index[i+1]
    i_str=str(i)
    if n - 10 > m + 4:
        for j in range(m + 4, n - 10):
            j_index.append(j)
            ii_index.append(i)

dataframe= dataframe.drop(columns=['name', 'date', 'speed', 'direct'])
essemble_test=[]
for i in range(len(i_index)):
    m=i_index[i]+1
    if i ==len(i_index)-1:
        n=len(dataframe)
    else:
        n=i_index[i+1]
    data=dataframe[m:n]
    dataframe_track=pd.DataFrame(data)
    essemble=shift(dataframe_track,60)
    essemble_values=essemble.values
    essemble_values=essemble_values.reshape(essemble_values.shape[0],15,4)
    essemble_values_new=essemble_values[:,:,:]
    essemble_values_new=essemble_values_new.reshape(essemble_values_new.shape[0],60)
    essemble_list=essemble_values_new.tolist()
    dif_data=dis_direct(essemble_list,60)
    essemble_test.append(dif_data)
essemble_ds=reduce(operator.add, essemble_test)
essemble_ds=np.stack(essemble_ds)
essemble_ds_new=essemble_ds
print(essemble_ds_new.shape)

# print(essemble_ds_new[0,152:200])
data_x=essemble_ds_new[:,:116]
data_y=essemble_ds_new[:,116:]

data_x_track=data_x.reshape((data_x.shape[0], 4, 29))
data_y_track=data_y.reshape((data_y.shape[0], 10, 4))
data_track=data_y_track[:,9:10,0:4]
data_track=data_track.reshape((data_track.shape[0],4))

data_track_16 = data_y_track[:,0:4,:].reshape(data_y_track.shape[0], 16)
x_scaler= MinMaxScaler(feature_range=(0, 1)).fit(data_x)
y_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(data_track)
y_scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(data_track_16)
data_x_new = x_scaler.transform(data_x)
data_track_new = y_scaler1.transform(data_track)

# print(data_track_new.data_max_)
# print(data_track_new.data_min_)
x_data=data_x_new.reshape((data_x_new.shape[0], 4, 29)).astype(np.float32)
# n1=int(len(x_data)*0.9)
# n1=19233

n1=4568

drop_x=pd.read_csv(f"./drop_idx_60.txt",names=['1'])
drop_idx=drop_x['1'].values
n1=n1-408
data_x_track_xin=np.delete(data_x_track,drop_idx,0)
x_data_xin=np.delete(x_data,drop_idx,0)
data_track_new_xin=np.delete(data_track_new,drop_idx,0)

train_track=data_track_new_xin[:n1,:]
valid_track=data_track_new_xin[n1:,:]
lat0=data_x_track_xin[n1:,3,0]
lon0=data_x_track_xin[n1:,3,1]
train_X = x_data_xin[:n1]
valid_X = x_data_xin[n1:]


j_index=[]
ii_index=[]
for i in range(len(i_index)):
    index=i_index[i]
    if i==len(i_index)-1:
        m=index+1
        n=len(dataframe)
    else:
        m=index+1
        n=i_index[i+1]
    i_str=str(i)
    if n-10>m+4:
        for j in range(m+4,n-10):
            j_index.append(j)
            ii_index.append(i)


j_names=j_index
i_names=ii_index
slp_label_ds=[]
uv_label_ds=[]
# 48 1336
glob_path=glob.glob('../compose_slp/compose_slp_npy_60_usa_2010_2024/*')
print(len(glob_path))
slp_label_latlon_ds=[]
for idx in range(len(j_names)):
    j = j_names[idx]
    i = i_names[idx]
    slp_label_66=np.load(f"../compose_slp/compose_slp_npy_60_usa_2010_2024/{str(i)}_{str(j)}_slp.npy")
    slp_label_latlon=np.load(f"../compose_slp/compose_slp_latlon_npy_usa_2010_2024_60/{str(i)}_{str(j)}_slp_latlon.npy")
    slp_label=slp_label_66
    slp_label_ds.append(slp_label)
    slp_label_latlon_ds.append(slp_label_latlon)
SLP_label=np.stack(slp_label_ds)
SLP_label_latlon=np.stack(slp_label_latlon_ds)
train_X_SLP_latlon=SLP_label_latlon.reshape(SLP_label_latlon.shape[0],81,81,2)
train_X_SLP=SLP_label.reshape(SLP_label.shape[0],1,81,81)
train_X_SLP=train_X_SLP.transpose(0,2,3,1)
train_SLP=np.concatenate([train_X_SLP[:,:,:,:],train_X_SLP_latlon[:,:,:,:]],axis=-1)
train_SLP_xin=np.delete(train_SLP,drop_idx,0)
train_P_label=train_SLP_xin[:n1]
valid_P_label=train_SLP_xin[n1:]


slp_tc_24=pd.read_csv(f"./slp_2010_2024_60.csv",index_col=0,header=0)

slp_tc=slp_tc_24.values
print(len(slp_tc))
y_scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(slp_tc)
slp_tc_new= y_scaler2.transform(slp_tc)
slp_tc_new_xin0=slp_tc_new.reshape(slp_tc_new.shape[0],10,7)
slp_tc_new_xin1=slp_tc_new_xin0[:,:,0:5]
slp_tc_new_xin2=slp_tc_new_xin1.reshape(slp_tc_new_xin1.shape[0],50)
slp_tc_new_xin0=np.delete(slp_tc_new_xin2,drop_idx,0)
fore_tc_train=slp_tc_new_xin0[:n1]
fore_tc_valid=slp_tc_new_xin0[n1:]
print(len(fore_tc_valid))
def scheduler(epoch):
   #    # 每隔50个epoch，学习率减小为原来的1/10
    if (epoch-10) % 5== 0 and epoch >9:
        lr = K.get_value(model.optimizer.lr)
        if lr>1e-6:
            K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

reduce_lr=LearningRateScheduler(scheduler)
seq_len=4
feature_num=29
slp_input = Input(shape=(81,81,3), name='slp_input')
slp_out = Conv2D(8, kernel_size=(7,7),strides=2,activation='relu')(slp_input)
slp_out = BatchNormalization()(slp_out)
slp_out = MaxPool2D(pool_size=(5,5), strides=4, padding='valid')(slp_out)
slp=Flatten()(slp_out)
slp=Dense(128,activation='relu')(slp)
slp=Dense(16,activation='relu')(slp)

gru_input = Input(shape=(seq_len,feature_num), name='gru_input')
encoder_outputs1=GRU(64,activation='relu', return_sequences=False,name="encode_lstm1")(gru_input)
encoder_outputs1=Dense(8,activation='relu')(encoder_outputs1)

fore_input = Input(shape=50, name='fore_input')
fore_output=Dense(128,activation='relu')(fore_input)
fore_output=Dense(64,activation='relu')(fore_output)
out_tracks=tf.concat([encoder_outputs1,fore_output],axis=-1)
out_tracks=Dense(16,activation='relu')(out_tracks)

query=Reshape((16, 1))(slp)
key=Reshape((1, 16))(out_tracks)
query_new=tf.repeat(query, repeats=[16], axis=2)
key_new=tf.repeat(key, repeats=[16], axis=1)
value=tf.add(query_new,key_new)
score = tf.matmul(query, key)
scaled_score = score / tf.math.sqrt(tf.cast(64,score.dtype))
weights=scaled_score
y = tf.multiply(weights, value)
out=Flatten()(y)

out_put=Dense(4,activation='relu')(out)

model = Model(inputs=[slp_input,gru_input,fore_input], outputs=out_put)
lr=1e-3
optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy', 'mae', 'mape'])
save_to = './GRU_CNN_60_mha.hdf5'
es=tf.keras.callbacks.ModelCheckpoint(save_to,monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.summary()
# history = model.fit([fore_tc_train],train_track,validation_data=([fore_tc_valid],valid_track) ,epochs=60,batch_size=16, callbacks=[es,reduce_lr],verbose=2,workers=5)
history = model.fit([train_P_label,train_X,fore_tc_train],train_track,validation_data=([valid_P_label,valid_X,fore_tc_valid],valid_track) ,epochs=60,batch_size=16, callbacks=[es,reduce_lr],verbose=2,workers=5)
model.load_weights('./GRU_CNN_60_mha.hdf5')
tf.compat.v1.enable_eager_execution()
yhat=model([valid_P_label,valid_X,fore_tc_valid])
def compute_dis(output_track, valid_target_y):
    output_track=output_track.numpy()
    yhat = output_track.reshape(output_track.shape[0], 4)
    valid_y = valid_target_y.reshape(valid_target_y.shape[0], 4)
    pre_yhat = y_scaler1.inverse_transform(yhat)
    act_y = y_scaler1.inverse_transform(valid_y)
    pre_yhat = pre_yhat.reshape(pre_yhat.shape[0],1, 4)[:,:,2:4]
    act_y = act_y.reshape(act_y.shape[0], 1,4)[:,:,2:4]
    lon_ds=[]
    # print(pre_yhat,act_y)
    dis_all_all = []
    for i in range(len(pre_yhat)):
        dis_all_ds = []
        for j in range(1):
            lat_pred_all = lat0[i]+pre_yhat[i, j, 0]
            lon_pred_all = lon0[i]+pre_yhat[i, j, 1]
            lat_true = lat0[i]+act_y[i, j, 0]
            lon_true = lon0[i]+act_y[i, j, 1]
            dis = getDistance(lat_pred_all, lon_pred_all, lat_true, lon_true)
            dis_all_ds.append(dis)
        dis_all_all.append(dis_all_ds)

    return dis_all_all

dis_all_all=compute_dis(yhat, valid_track)
print('dis_error:',np.mean(dis_all_all,axis=0))
#  286[160.12881089]
# output_track=yhat.numpy()
# yhat = output_track.reshape(output_track.shape[0], 8)
# valid_y = valid_track.reshape(valid_track.shape[0], 8)
# pre_yhat = y_scaler.inverse_transform(yhat)
# act_y = y_scaler.inverse_transform(valid_y)
# pre_yhat = pre_yhat.reshape(pre_yhat.shape[0],4, 2)
# act_y = act_y.reshape(act_y.shape[0], 4, 2)
#
# pred_lat=pre_yhat[:,:,0]+lat0.reshape(-1,1).repeat(4,axis=-1)
# pred_lon=pre_yhat[:,:,1]+lon0.reshape(-1,1).repeat(4,axis=-1)
#
# pred_lat = pred_lat.reshape(pred_lat.shape[0],4, 1)
# pred_lon =pred_lon.reshape(pred_lon.shape[0], 4, 1)
# pred_value=np.concatenate([pred_lat,pred_lon],axis=-1)
# print(pred_value.shape)
# pred_value=pred_value.reshape(pred_value.shape[0],8)
#
# truth_lat=act_y[:,:,0]+lat0.reshape(-1,1).repeat(4,axis=-1)
# truth_lon=act_y[:,:,1]+lon0.reshape(-1,1).repeat(4,axis=-1)
# truth_lat = truth_lat.reshape(truth_lat.shape[0],4, 1)
# truth_lon =truth_lon.reshape(truth_lon.shape[0], 4, 1)
# truth_value=np.concatenate([truth_lat,truth_lon],axis=-1)
# truth_value=truth_value.reshape(truth_value.shape[0],8)
#
# import csv
# def essemble_path(name,result):
#     path='./'+name
#     with open(path, 'w', newline='') as csvfile:
#         writer  = csv.writer(csvfile)
#         for row in result:
#             writer.writerow(row)

# essemble_path('result_track_fore24.csv',pred_value)
# essemble_path('result_track_truth.csv',truth_value)
