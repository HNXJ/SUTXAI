#!/usr/bin/env python
import numpy as np
import PhysionetSet
# from get_12ECG_features import get_12ECG_features


def label_threshold(x, th=0.75):
    
    m = np.max(x)*th
    y = x > m
    return y.astype(np.int16)
    

def run_12ECG_classifier(data, header_data, classes, model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    # features = get_12ECG_features(data,header_data)
    
    try:
        features = np.reshape(data[:, :4000], [1, 12, -1, 1])
    except:
        x = np.zeros(1, 12, 4000, 1)
        x[:, :, :features.shape[1], :] = features
        features = np.reshape(x[:, :4000], [1, 12, -1, 1])
        
    score = model(features)
    label = label_threshold(score.numpy())
    # print(score.shape)
    # print(label.shape)
    current_label = np.reshape(label, [9])

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])
        
    # print('->', current_label, '\n->', current_score)
    return current_label, current_score

def load_12ECG_model(filename=''):
    
    loaded_model = PhysionetSet.load_model_weight(path=filename)
    return loaded_model
