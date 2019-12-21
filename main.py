 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
os.makedirs('figs', exist_ok=True)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import datetime
from keras.models import load_model

'''Usage
>>>python main.py -gi 0 -id Exp_BlockNum1 -bn 2 -abn
'''

'''Argparse'''
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-id', '--script-id', type=str, default='Exp_test', help="ID of this experiment")
parser.add_argument('-gi', '--gpu-id', type=int, default=0, help="ID of GPU card to use")
parser.add_argument('-bn', '--block-num', type=int, default=2, help="block number of CNN model")
parser.add_argument('-abn', '--is-batch-normalization', action='store_true', help="whether to add batch normalization layer after conv")
parser.add_argument('-fn', '--block-filter-num', type=int, default=32, help="filter number of CNN model")
parser.add_argument('-ks', '--block-kernel-size', type=int, default=5, help="kernel size of CNN model")
parser.add_argument('-i', '--initializers', type=str, default='random_normal', help="block number of CNN model")
parser.add_argument('-b', '--batch-size', type=int, default=64, help="training batch size")
parser.add_argument('-cn', '--class-number', type=int, default=2, help="Number of classes")
parser.add_argument('-e', '--epochs', type=int, default=30, help="Training epochs")
parser.add_argument('-p', '--history-file', type=str, default = "training_history.json", help="Training history")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

from utils import load_data, plot_confusion_matrix, plot_history
from models import get_CNN_model, get_data_generator, get_callback

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
  
import logging
'''Logging Setup '''
os.makedirs('log', exist_ok=True)

logging.basicConfig(filename='log/%s.log'%(datetime.datetime.now().strftime(args.script_id + "--%Y-%m-%d-%H_%M_%S")) ,
                            filemode='a',
                            format='[%(levelname)s] %(asctime)s | %(filename)s | %(funcName)s | %(lineno)d : %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)

global logger
logger = logging.getLogger('Minist classification configuration')
logger.info('Start up Script')

# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(args)

epochs = args.epochs 
batch_size = args.batch_size

'''Load dataset'''
X_train, X_val, Y_train, Y_val, X_test = load_data()

'''Load CNN model'''
model = get_CNN_model(block_num=args.block_num, block_filter_num=args.block_filter_num, \
    block_kernel_size=args.block_kernel_size, initializers = args.initializers, is_batch_normalization = args.is_batch_normalization)
 
'''Train on training dataset'''
# Data augmentation to prevent overfitting
datagen = get_data_generator(X_train)
 
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks= get_callback())
# Plot the loss and accuracy curves for training and validation 
plot_history(history)
plt.show()
plt.savefig('figs/history-' + args.script_id + '.png')

'''Predict on validation dataset'''
# Load best model
model = load_model("models/weights.h5")
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
cm = confusion_matrix(Y_true, Y_pred_classes) 
logger.info('Confusion matrix')
print(cm)
# plot the confusion matrix
# plot_confusion_matrix(cm, classes = range(10)) 
plot_confusion_matrix(Y_true, Y_pred_classes, classes = range(10)) 
 
plt.show()
plt.savefig('figs/cm-' + args.script_id + '.png')
val_acc = accuracy_score(Y_true, Y_pred_classes)
logger.info('Accuarcy on val dataset is: %.4f'%val_acc)



'''Predict on testset'''
# predict results
prediction = model.predict(X_test)
# select the indix with the maximum probability
results = np.argmax(prediction,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results], axis = 1)
# Save to csv file
submission.to_csv("cnn_mnist_datagen.csv",index=False)
 