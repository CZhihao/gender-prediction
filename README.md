# gender-prediction
Gender prediction by simple 7 layer CNN

Requirement of libraries:
Keras (I tested with GPU support, CPU also should work without bug)
Thensorflow
Numpy
Python3 (Python2 not tested, possibly need minor modification. But we shouldn’t use python2 any more in 2018)

Following are optional:
opencv3
matplotlib

1. First of all, prepare your dataset as following structure: root_project/all,  root_project/train, root_project/val. all means whole dataset, train means dataset for training, val means dataset for validation and evaluation. Then in each dataset create 2 sub folder ‘man’ and ‘woman’ and put the corresponding images in each folder.

2. (Optional) If you want, you can make a table of ground truth, by running
	python3 dataset.py
You will find a txt and npy file in the folder ‘all’. But it is not required for training or inference.

3. Traning. Run
	python3 gender_train.py
You will get the saved model file (model.h5) and weight file (weightsxx.h5) at the project root folder.
In the code file gender_train.py, you can modify the epoch and batch size at line 19. Batch size =32 can get a good result. If have bug report of memory, please check this parameter.

4. Prediction on single image. Please run gender_single_predict -image_path -weight_path. For ex,
	python3 gender_single_predict.py test1.jpg weights70.h5

5. Prediction on a group of images and calculate the mAP, run python3 gender_mAP.py -dataset_path -weight_path
	python3 gender_mAP.py val weights70.h5
