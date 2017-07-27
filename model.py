import csv
import numpy as np
import cv2
import keras as ks

lines = []
file_directory = "/Users/hemanth/Udacity/behavioralData/data/"
with open(file_directory+"driving_log.csv") as csvfile:
    headers=1
    itr=0
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if(itr>headers-1):
            lines.append(row)
        itr+=1

print("hi")

Split_data=0.8
np.random.shuffle(lines)
train_lines = lines[:int(0.8*len(lines))]
test_lines = lines[int(0.8*len(lines)):]

def generateData(file_directory,train_lines,batch_size =1):
    while(1):
        np.random.shuffle(train_lines)
        for batch_itr in range(0,int(len(train_lines)/batch_size)):
            image_data=[]
            steering_data=[]
            if(batch_itr!=int(len(train_lines)/batch_size)-1):
                cur_itr_start = batch_itr*batch_size
                cur_itr_end = cur_itr_start+batch_size
            else:
                cur_itr_start = batch_itr*batch_size
                cur_itr_end = len(train_lines)
            for itr in range(cur_itr_start,cur_itr_end):
                image = cv2.imread(file_directory+train_lines[itr][0])
                image_data.append(image)
                steering_data.append(float(train_lines[itr][3]))
                yield (np.array(image_data),np.array(steering_data))


batch_size1 = 1
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Activation
from keras.layers.convolutional import Convolution2D,Conv2D
from keras.layers.pooling import MaxPooling2D
#from keras import losses
rows,col,channels=160,320,3
model = Sequential()
#read a 320x160x3 image
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(rows,col,channels),output_shape=(rows,col,channels)))
#model.add(Flatten(input_shape= (rows,col,channels)))
# Feature Map of shape 316x156x6
model.add(Conv2D(6,5,1))
# Feature Map of shape 316x156x6
model.add(Activation('relu'))

# Feature Map of shape 158x78x6
#using keras 1 not keras 2, SO has to use border_mode instead of padding
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same"))
# Feature Map of shape 79x39x16
model.add(Conv2D(16,5,1))
# Feature Map of shape 79x39x16
model.add(Activation('relu'))
# Feature Map of shape 39x19x16
#using keras 1 not keras 2, SO has to use border_mode instead of padding
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same"))
# Feature Map of shape 11856

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))
#adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer='adam')
model.fit_generator(generateData(file_directory,train_lines,batch_size = 1), int((len(train_lines))/batch_size1), 2,1,None,generateData(file_directory,test_lines,batch_size = 1),int((len(test_lines))/batch_size1) )

model.save('model.h5')
print('model saved')

'''
for i,j in generateData(file_directory,lines):
    print(len(i))
    cv2.imshow("hi",i[0])
    cv2.waitKey()


'''
