from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import sys
import cv2


img_width, img_height = 150, 150
try:
    path_image=sys.argv[1]
except IndexError:
    print("\nImage path not given\n")
try:
    path_weight=sys.argv[2]
except IndexError:
    print("\nWeight path not given\n")

model=load_model('model.h5')
model.load_weights(path_weight)


#single image prediction




img = load_img(path_image,False,target_size=(img_width,img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

#make the prediction of class and its probability 
preds = model.predict_classes(x)
probs = model.predict_proba(x)
print(preds, probs)

#Draw prediction on the image
predicted_img=cv2.imread(path_image)
if preds[0,0]==0:
    cv2.putText(predicted_img,'man',(10,30), cv2.FONT_HERSHEY_COMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
else:
    cv2.putText(predicted_img,'woman',(10,30), cv2.FONT_HERSHEY_COMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('result',predicted_img)
cv2.waitKey(0)
#close the image window to finish the program
cv2.destroyAllWindows()
K.clear_session()

