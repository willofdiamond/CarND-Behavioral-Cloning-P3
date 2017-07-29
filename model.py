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
    correction = 0.3
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
                image_left = cv2.imread(file_directory+"IMG/"+train_lines[itr][1].split('/')[-1])
                image_right = cv2.imread(file_directory+"IMG/"+train_lines[itr][2].split('/')[-1])
                #print(file_directory+"IMG/"+train_lines[itr][1].split('/')[-1])
                steering=float(train_lines[itr][3])
                steering_left = steering+correction
                steering_right = steering-correction
                image_flipped=np.fliplr(image)
                '''
                image_data.append(image)
                #flipping the image to train car for right curve turns
                image_data.append(image_flipped)
                #Add left side image
                image_data.append(image_left)
                #Add right side image
                image_data.append(image_right)
                '''
                image_data.extend([image,image_flipped,image_left,image_right])
                steering_flipped=-1*steering
                '''
                steering_data.append(steering)
                steering_data.append(steering_flipped)
                steering_data.append(steering_left)
                steering_data.append(steering_right)
                '''
                steering_data.extend([steering,steering_flipped,steering_left,steering_right])

                yield (np.array(image_data),np.array(steering_data))


batch_size1 = 1
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Lambda,Activation
from keras.layers.convolutional import Convolution2D,Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras import losses
rows,col,channels=160,320,3
 # NVIDIA Architecture
model = Sequential()
#read a  3@160x320 input planes
#Crop the image to eliminate the other objects
model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(rows,col,channels)))
# Feature Map of shape 3@90x320
rows,col,channels=90,320,3
# Normalize the data
model.add(Lambda(lambda x: x/127.5 - 1.,output_shape=(rows,col,channels)))

# Convolution layer 1
model.add(Conv2D(24,5,2))
# Feature map 24@43x158
model.add(Activation('relu'))
# Feature map 24@43x158
# Convolution layer 2
model.add(Conv2D(36,5,2))
# Feature Map of shape 36@20x77
model.add(Activation('relu'))
# Feature Map of shape 36@20x77
# Convolution layer 3
model.add(Conv2D(48,5,2))
# Feature Map of shape 48@8x37
model.add(Activation('relu'))
# Feature Map of shape 48@8x37
# Convolution layer 4
model.add(Conv2D(64,3,1))
# Feature Map of shape 64@6x35
model.add(Activation('relu'))
# Feature Map of shape 64@6x35
# Convolution layer 5
model.add(Conv2D(64,3,1))
# Feature Map of shape 64@4x33
model.add(Activation('relu'))
# Feature Map of shape 64@4x33
# Flatten the planes
model.add(Flatten())
# Feature Map of shape 8448
model.add(Dense(100))
# Feature Map of shape 100
model.add(Activation('relu'))
model.add(Dense(50))
# Feature Map of shape 50
model.add(Activation('relu'))
model.add(Dense(10))
# Feature Map of shape 10
model.add(Activation('relu'))
model.add(Dense(1))


#optimizing the loss functon
#adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer='adam')
model.fit_generator(generateData(file_directory,train_lines,batch_size = 1), int((4*len(train_lines))/batch_size1), 2,1,None,generateData(file_directory,test_lines,batch_size = 1),int((4*len(test_lines))/batch_size1) )

model.save('model.h5')
print('model saved')

'''
for i,j in generateData(file_directory,lines):
    print(len(i))
    cv2.imshow("hi",i[0])
    cv2.waitKey()


'''
