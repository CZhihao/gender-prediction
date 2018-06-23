from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import sys



img_width, img_height = 150, 150

#Evaluation images directory: 'all' ,'train' or 'dir'
try:
    dir_evaluation=sys.argv[1]
except IndexError:
    print("\nImage path not given\n")
#Weight file path    
try:
    file_weight=sys.argv[2]
except IndexError:
    print("\nWeight file path not given\n")    


model=load_model('model.h5')
model.load_weights(file_weight)

#Argument 1: path of the test image directory 
#calculate mAP with test data generator

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        dir_evaluation,
        target_size=(img_width, img_height),
        shuffle = False,
        class_mode=None,
        batch_size=1)


filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)
predict=predict.reshape(nb_samples,)

#Calculate de difference value between prediction and ground truth
notes=np.subtract(predict,test_generator.classes)

bingo=0
for note in notes :
    if abs(note)<0.5:
        bingo+=1
print('*****************************************************************')
print('mAP report')
print('Evaluation dataset: '+dir_evaluation)
print (str(bingo)+ ' images correctly predicted among '+str(nb_samples)+' evaluation images' )
mAP=bingo/nb_samples*100
print('mAP(%%): %0.2f'%mAP) 
print('*****************************************************************')       


K.clear_session()
