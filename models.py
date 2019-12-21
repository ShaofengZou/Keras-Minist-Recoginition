
import logging
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Set the CNN model 
# default CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
# 'random_normal' 'orthogonal' 'random_uniform'
def get_CNN_model(block_num = 2, block_filter_num = 32, block_kernel_size = 5, initializers = 'random_normal', is_batch_normalization = False, filter_second_block = 64, kernel_second_block = 3, ):
    logger = logging.getLogger('Minist classification configuration.models.py')

    model = Sequential()

    assert block_num > 0, "Oh no! The number of block should bigger than 0"

    for _ in range(block_num):
        model.add(Conv2D(filters = block_filter_num, kernel_size = (block_kernel_size,block_kernel_size),padding = 'Same',
                        kernel_initializer = initializers, 
                        input_shape = (28,28,1)))
        if is_batch_normalization == True:
            model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2D(filters = block_filter_num, kernel_size = (block_kernel_size,block_kernel_size),padding = 'Same', 
                        kernel_initializer = initializers))
        if is_batch_normalization == True:
            model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

    # model.add(Conv2D(filters = filter_second_block, kernel_size = (kernel_second_block,kernel_second_block),padding = 'Same', 
    #                 activation ='relu'))
    # model.add(Conv2D(filters = filter_second_block, kernel_size = (kernel_second_block,kernel_second_block),padding = 'Same', 
    #                 activation ='relu'))
    # model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    # model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    logger.info(model.summary())
    return model

def get_data_generator(X_train):
    # With data augmentation to prevent overfitting  
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)
    return datagen

def get_callback():
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

    os.makedirs('models', exist_ok = True)
    # Save the model after every epoch, the latest best model according to the quantity monitored will not be overwritten.
    checkpointer = ModelCheckpoint(filepath="models/weights.h5", monitor='val_loss', verbose=1, save_best_only=True)
    # Stop training when a monitored quantity has stopped improving.
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    return [learning_rate_reduction, checkpointer, early_stop]
