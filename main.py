from get_12ECG_features import get_12ECG_features
from run_12ECG_classifier import *
import tensorflow as tf
import PhysionetSet
import numpy as np
import driver
import os


def test_run():
    f = "A0001.mat"
    input_directory = "xtemp/"
    print('    {}/{}...'.format(5, 10))
    tmp_input_file = os.path.join(input_directory, f)
    data, header_data = driver.load_challenge_data(tmp_input_file)
    # driver.run()
    features=np.asarray(get_12ECG_features(data,header_data))
    return data


def train(model, xtrain, ytrain, epochs=10, batch_size=50, valid_split=0.15, lr=0.01, logdir="logs"):
    
    # log_dir= logdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
    # tensorboard_cb = TensorBoard(log_dir=log_dir, update_freq='batch', histogram_freq=1) 
    
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adagrad(learning_rate=lr),
       metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    
    model.fit(x=xtrain, y=ytrain, validation_split=valid_split,
              batch_size=batch_size, epochs=epochs, shuffle=True)
                # , callbacks=[tensorboard_cb])
    
    
def run1():
    
    id1 = 1
    id2 = 5001# ECG_SHAPE = (12, 4000, 1)
    
    dataset = PhysionetSet.PhysionetSet(foldername="Data/")
    dataset.load_data(id1=id1, id2=id2)
    dataset.get_save_data(id1=id1, id2=id2, valid_ratio=1.0, tsize=4000)
    dataset.load_train_data(scaler=20)
    
    print(dataset.xtrain.shape)
    print(dataset.ytrain.shape)
    print(dataset.xtrain[11, 1:12, 1, :])
    print(dataset.ytrain[11, :])
    
    
def run2(ide=10):
    
    dataset = PhysionetSet.PhysionetSet(foldername="Data/")
    dataset.load_train_data(scaler=20)
    
    model1 = PhysionetSet.load_model_weight(path='Trained/CNNP1/m2.ckpt')
    print(model1(dataset.xtrain[ide:ide+1, :, :, :]))
    print(label_threshold(model1(dataset.xtrain[1:2, :, :, :]), th=0.8))
    return model1, dataset
    
    
# model1, dataset1 = run2()
run2(34)