import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

path = r'./data/01-12' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
fields = ['Protocol', 'Destination Port','ACK Flag Count','Init_Win_bytes_forward','min_seg_size_forward',
'Fwd IAT Mean','Fwd IAT Max','Packet Length Std','Max Packet Length','Average Packet Size',
'Min Packet Length','Flow Duration','Flow IAT Mean','Flow IAT Max','Subflow Fwd Bytes','Label']
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0,usecols=fields,skipinitialspace=True)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

print("DONE LOADING TRAIN DATA")

x = frame.iloc[:,0:-1]  #independent columns
y = frame.iloc[:,-1]    #target column

encoder_y = LabelBinarizer()
encoder_y.fit(y)

#y[y == "BENIGN"] = 0
#y[y != "BENIGN"] = 1
#yp = y.to_numpy()
#class_weights = class_weight.compute_class_weight('balanced', np.unique(yp), yp)
#class_weight_dict = dict(enumerate(class_weights))                                  
yp = encoder_y.transform(y)


preprocess = make_column_transformer(
    (OrdinalEncoder(),['Protocol', 'Destination Port']),
    (MinMaxScaler(),['ACK Flag Count','Init_Win_bytes_forward','min_seg_size_forward',
'Fwd IAT Mean','Fwd IAT Max','Packet Length Std','Max Packet Length','Average Packet Size',
'Min Packet Length','Flow Duration','Flow IAT Mean','Flow IAT Max','Subflow Fwd Bytes']))

xp = preprocess.fit_transform(x)

print("DONE PREPROCESS TRAIN DATA")


input_1 = keras.layers.Input((1,))
input_2 = keras.layers.Input((1,))
input_3 = keras.layers.Input((13,))

input_1_emb = keras.layers.Embedding(3, 2,  mask_zero=False)(input_1)
input_1_emb = keras.layers.Flatten()(input_1_emb)

input_2_emb = keras.layers.Embedding(65535, 50,  mask_zero=False)(input_2)
input_2_emb = keras.layers.Flatten()(input_2_emb)

outputs = keras.layers.Concatenate(axis=-1)([input_1_emb,input_2_emb,input_3])
outputs = keras.layers.BatchNormalization()(outputs)
outputs = keras.layers.Dense(128, activation='relu')(outputs)
outputs = keras.layers.Dropout(0.1)(outputs)
#outputs = keras.layers.Dense(128, activation='relu')(outputs)
outputs = keras.layers.Dense(64, activation='relu')(outputs)

outputs = keras.layers.Dense(13,activation='softmax')(outputs)
model = keras.models.Model(inputs=[input_1,input_2,input_3], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("MODEL BUILT, START TRAINING NOW")

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights_only= True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)

history = model.fit(x=[xp[:,0], xp[:,1], xp[:,2:]], y=yp, epochs=100, batch_size=2048,validation_split=0.2,
shuffle=True,use_multiprocessing=True,callbacks=[es,mc])

print("DONE TRAINING, SAVING TRAINING HISTORY")

with open('./trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("LOADING TEST DATA")
path = r'./data/03-11' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
fields = ['Protocol', 'Destination Port','ACK Flag Count','Init_Win_bytes_forward','min_seg_size_forward',
'Fwd IAT Mean','Fwd IAT Max','Packet Length Std','Max Packet Length','Average Packet Size',
'Min Packet Length','Flow Duration','Flow IAT Mean','Flow IAT Max','Subflow Fwd Bytes','Label']
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0,usecols=fields,skipinitialspace=True)
    li.append(df)

frame2 = pd.concat(li, axis=0, ignore_index=True)
print("DONE LOADING TEST DATA")
xpt = preprocess.fit_transform(frame2.iloc[:,0:-1])
print("START PREDICTION ON TEST DATA")
prediction = model.predict([xpt[:,0], xpt[:,1], xpt[:,2:]],batch_size=2048)
print("DONE PREDICTION")
pred_label = encoder_y.inverse_transform(prediction)

print("CALCULATING SCORES")
yt = frame2.iloc[:,-1].to_numpy()
prc, recall, f1, suport = precision_recall_fscore_support(yt, pred_label, average='weighted')
acc = accuracy_score(yt, pred_label)

with open('./testResults', 'w') as file_pi:
    file_pi.writelines("Precisiion \n" +str(prc) + "\n" )
    file_pi.writelines("Recall \n"+ str(recall) + "\n")
    file_pi.writelines("F1 \n" + str(f1) + "\n")
    file_pi.writelines("Accuracy" + str(acc) + "\n")

prc, recall, f1, suport = precision_recall_fscore_support(yt, pred_label, average='macro')
with open('./testResultsMacro', 'w') as file_pi:
    file_pi.writelines("Precisiion \n" +str(prc) + "\n" )
    file_pi.writelines("Recall \n"+ str(recall) + "\n")
    file_pi.writelines("F1 \n" + str(f1) + "\n")
    file_pi.writelines("Accuracy" + str(acc) + "\n")

np.save("truetest.npy", yt)
np.save("predtest.npy", pred_label)