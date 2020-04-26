from tensorflow.python.client import device_lib
import tensorflow as tf
import scipy.io as sio
import numpy as np
# import zipfile 
import os


# def zipdir(path, ziph):
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             ziph.write(os.path.join(root, file))


# def undzip(path):
#     folder_path = '/content/Model_ext'
#     with zipfile.ZipFile(path, 'r') as zip_ref:
#         zip_ref.extractall(folder_path)


# def compress_model(modeldir='Trained/'):
#     z = zipfile.ZipFile('Model.zip', 'w')
#     zipdir(modeldir, z)
#     z.close()


# def save_model_weights(model, model_name="NN", epochs=1):
#     model.save_weights("Trained/" + model_name + "/m{epoch:01d}.ckpt".format(epoch=epochs))
#     compress_model('Trained/')
#     return


def load_model_weight(path='Trained/CNNP1/m1.ckpt', ECG_SHAPE=(12, 4000, 1)):
    model = Encoder1(shape=ECG_SHAPE)
    model.load_weights(path)
    return model


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    

lst = get_available_gpus()
if '/device:GPU:0' in lst:
    tf.device('/device:GPU:0')
    print('GPU is activated')
elif '/device:XLA_CPU:0' in lst:
    tf.device('/device:XLA_CPU:0')
    print('TPU is activated')
else:
    print('CPU only available')
    
    
class PhysionetSet:
    
    def __init__(self, foldername="Data/"):
        
        self.leads = dict()
        self.specs = dict()
        self.labels = dict()
        self.diseases = dict()
         
        self.diseases['AF'] = 0
        self.diseases['I-AVB'] = 1
        self.diseases['LBBB'] = 2
        self.diseases['Normal'] = 3
        self.diseases['PAC'] = 4
        self.diseases['PVC'] = 5
        self.diseases['RBBB'] = 6
        self.diseases['STD'] = 7
        self.diseases['STE'] = 8

        self.diseases[0] = 'AF'
        self.diseases[1] = 'I-AVB'
        self.diseases[2] = 'LBBB'
        self.diseases[3] = 'Normal'
        self.diseases[4] = 'PAC'
        self.diseases[5] = 'PVC'
        self.diseases[6] = 'RBBB'
        self.diseases[7] = 'STD'
        self.diseases[8] = 'STE'

        self.xtrain = None
        self.ytrain = None
        self.xvalid = None
        self.yvalid = None
        
        self.foldername = foldername       
        return
        
    
    @staticmethod
    def vstr(a):
        s = str(a)
        t = '0'*(4 - len(s))
        s = t + s
        return s

    def disease(self, x, th=0.5):

        m = np.max(x)*th
        s = ""
        for i in range(9):
            if x[i] > m:
                s += self.diseases[i] + ", "
        return s
    
    def load_data(self, id1=4762, id2=5290):
        
        for i in range(id1, id2):
            if i % 1000 == 0:
                print(i, ' of ', (id2 - id1 + 1))
                
            mat = sio.loadmat(self.foldername + 'A' + self.vstr(i) + '.mat')
            hea = open(self.foldername + 'A' + self.vstr(i) + ".hea", 'r+')

            s = hea.readlines()
            d = s[15][:-1].split(' ')
            d = d[1:][0].split(',')
            
            self.leads[i] = mat['val']
            self.specs[i] = [item for item in d]
            
            label = np.zeros([1, 9])
            for item in d:
                label[0, self.diseases[item]] = 1
            self.labels[i] = label
            
        return
    
    def get_save_data(self, id1=4762, id2=5290, tsize=4000, valid_ratio=0.9):
        
        x0 = np.zeros([id2-id1, 12, tsize, 1])
        y0 = np.zeros([id2-id1, 9])
        
        for i in range(id1, id2):
            if i % 1000 == 0:
                print(i, ' of ', (id2 - id1 + 1))
            x = self.leads[i]
            x0[i-id1, :, :, :] = np.reshape(x[:, :tsize], [1, 12, tsize, 1])
            y0[i-id1, :] = self.labels[i]
            
        m = int(x0.shape[0]*valid_ratio)

        np.save('xt.npy', x0[:m+1])
        np.save('yt.npy', y0[:m+1])
        np.save('xv.npy', x0[m:])
        np.save('yv.npy', y0[m:])
        
        return
    
    def get_train_data(self, id1=4762, id2=5290, valid_ratio=0.9, tsize=4000, output=False): 
        
        x0 = np.zeros([id2-id1, 12, tsize, 1])
        y0 = np.zeros([id2-id1, 9])
        
        for i in range(id1, id2):
            if i % 100 == 0:
                print(i, ' of ', (id2 - id1 + 1))
            x = self.leads[i].astype(np.float32)
            x0[i-id1, :, :, :] = np.reshape(x[:, :tsize], [1, 12, tsize, 1])
            y0[i-id1, :] = self.labels[i].astype(np.float32)
            
        m = int(x0.shape[0]*valid_ratio)
        if output:
            return x0[:m], y0[:m], x0[m:], y0[m:]
        
        self.xtrain = x0[:m+1]/1000
        self.ytrain = y0[:m+1]
        self.xvalid = x0[m:]/1000
        self.yvalid = y0[m:]
        
        return
    
    def load_train_data(self, output=False, scaler=10):
        
        x0 = np.load('xt.npy').astype(np.float32)
        y0 = np.load('yt.npy').astype(np.float32)
        x1 = np.load('xv.npy').astype(np.float32)
        y1 = np.load('yv.npy').astype(np.float32)
        
        if output:
            return x0, y0, x1, y1
        
        self.xtrain = x0/scaler
        self.ytrain = y0
        self.xvalid = x1/scaler
        self.yvalid = y1
        
        return


class Encoder1(tf.keras.Sequential):
    def __init__(self, shape):
        super(Encoder1, self).__init__()
        self.add(tf.keras.layers.Conv2D(input_shape=shape, filters=4,
                                        kernel_size=(1, 100), strides=(1, 5),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 20), strides=(1, 2), padding='same'))

        self.add(tf.keras.layers.Conv2D(input_shape=(12, 400, 4), filters=13,
                                        kernel_size=(1, 30), strides=(1, 4),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 10), strides=(1, 2), padding='same'))

        self.add(tf.keras.layers.Conv2D(input_shape=(12, 50, 13), filters=50,
                                        kernel_size=(1, 5), strides=(1, 5),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 1), padding='same'))

        # self.add(tf.keras.layers.Conv2D(input_shape=(2, 3, 120), filters=1600,
        #                                 kernel_size=(2, 2), strides=(1, 1),
        #                                 padding='same', activation='relu'))
        # self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        
        self.add(tf.keras.layers.Flatten(input_shape=[12, 10, 50]))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(500, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(60, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(9, activation='sigmoid'))
        return
  

class Encoder2(tf.keras.Sequential):
    def __init__(self, shape):
        super(Encoder2, self).__init__()
        self.add(tf.keras.layers.Conv2D(input_shape=shape, filters=3,
                                        kernel_size=(1, 100), strides=(1, 5),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 20), strides=(1, 2), padding='same'))

        self.add(tf.keras.layers.Conv2D(input_shape=(12, 400, 4), filters=10,
                                        kernel_size=(1, 40), strides=(1, 5),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 10), strides=(1, 2), padding='same'))
        
        self.add(tf.keras.layers.Flatten(input_shape=[12, 40, 10]))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(500, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(60, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(9, activation='sigmoid'))
        return


class Encoder3(tf.keras.Sequential):
    def __init__(self, shape):
        super(Encoder3, self).__init__()
        self.add(tf.keras.layers.Conv2D(input_shape=shape, filters=4,
                                        kernel_size=(1, 100), strides=(1, 10),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.Dropout(rate=0.5))

        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 20), strides=(1, 4), padding='same'))

        self.add(tf.keras.layers.Conv2D(input_shape=(12, 400, 4), filters=16,
                                        kernel_size=(1, 40), strides=(1, 5),
                                        padding='same', activation='relu'))
        self.add(tf.keras.layers.MaxPool2D(pool_size=(1, 10), strides=(1, 2), padding='same'))
        self.add(tf.keras.layers.Dropout(rate=0.5))

        self.add(tf.keras.layers.Flatten(input_shape=[12, 10, 16]))
        self.add(tf.keras.layers.Dropout(rate=0.5))

        self.add(tf.keras.layers.Dense(1000, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(400, activation='sigmoid'))
        self.add(tf.keras.layers.Dropout(rate=0.5))
        self.add(tf.keras.layers.Dense(9, activation='sigmoid'))
        return

    
