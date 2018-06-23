import os 
import numpy as np


#this code can make a label table as the following structure:
''' 
   image  gender
 123456.jpg  man
 123457.jpg  woman
    ...      ...
    
''' 
#does this useful? I don't know. At least i didn't use this table for such a simple binary classification   
path_dataset='all'
#print(type(os.listdir(path_dataset+'/man')))
list_man=os.listdir(path_dataset+'/man')
list_woman=os.listdir(path_dataset+'/woman')
list_man=[[x]+[0] for x in list_man]
list_woman=[[x]+[1] for x in list_woman]


list_man.sort()
list_woman.sort()
whole_list=list_man+list_woman
whole_list.sort()
whole_list=np.array(whole_list)

#save the label table as the numpy npy format
np.save(path_dataset+'/label',whole_list)
#save the label table as the txt format
with open(path_dataset+'/label.txt','w')as txt_label:
    txt_label.write('  image     gender'+'\n')
    for row in whole_list:
        if row[1]=='0':
            txt_label.write(row[0]+'   man\n')
        else:
            txt_label.write(row[0]+'  woman\n')

