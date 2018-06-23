from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import os
import matplotlib.pyplot as plt


# dimensions of images.
img_width, img_height = 150, 150

train_data_dir = 'train'
validation_data_dir = 'val'
nb_train_samples = 500
nb_validation_samples = 75
epochs = 100
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# 7 layer simple CNN. Full connected layer at end to make the binary prediction
# add more convolutional and pooling layers could gain a better result
# other structures like residual network could work better than this tradional CNN 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# We consider that this is a task of binary prediction because of only 2 classes. 
# Class prediction is also possible.
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(    
    rescale=1./255,
    #shera mapping coefficient
    shear_range=0.2,
    #randomly zooming
    zoom_range=0.2,
    horizontal_flip=True,
    #filling om newly created pixels after rotation or shift
    fill_mode='nearest')

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# weight file will be saved at the end of every N epochs  
class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.epoch=0
        

    def on_batch_end(self, batch, logs={}):
        if (batch+1)==(nb_train_samples // batch_size):
            self.epoch+=1
        if self.epoch % self.N == 0 and self.epoch!=0:
            name = 'weights'+str(self.epoch)+'.h5' 
            self.model.save_weights(name)
       

history_callback=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[WeightsSaver(model, 10)],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
    
  
# save the loss and accuracy values of both training and validation 
# the log files are saved there:
path_history="history"
if not os.path.exists(path_history):
    os.makedirs(path_history)
    

loss_history = history_callback.history['loss']
numpy_loss_history = np.array(loss_history)
np.savetxt(path_history+"/loss_history.txt", numpy_loss_history, delimiter=",")

acc_history = history_callback.history['acc']
numpy_loss_history = np.array(acc_history)
np.savetxt(path_history+"/acc_history.txt", numpy_loss_history, delimiter=",")

val_loss_history = history_callback.history['val_loss']
numpy_loss_history = np.array(val_loss_history)
np.savetxt(path_history+"/val_loss_history.txt", numpy_loss_history, delimiter=",")

val_acc_history = history_callback.history['val_acc']
numpy_loss_history = np.array(val_acc_history)
np.savetxt(path_history+"/val_acc_history.txt", numpy_loss_history, delimiter=",")

# draw the loss/accuracy-epoch plot
plt.subplot(2,1,1)
plt.plot(loss_history) 
plt.plot(val_loss_history) 
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  


plt.subplot(2,1,2)
plt.plot(acc_history) 
plt.plot(val_acc_history) 
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show() 


model.save('model.h5')
K.clear_session()

