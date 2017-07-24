import csv
import numpy as np
import cv2

lines = []
file_directory = "D:/Udacity_selfdriving/data/data/"
with open(file_directory+"driving_log.csv") as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        lines.append(row)
        
print("hi")
        

    
def generateData(file_directory,batch_size,lines):
    images=[]
    steering=[]
    for batch_itr in range(int(len(lines)/batch_size)):
        if(batch_itr!=int(len(lines)/batch_size)-1):
            cur_itr_start = batch_itr*batch_size
            cur_itr_end = cur_itr_start+batch_size
        else:
            cur_itr_start = batch_itr*batch_size
            cur_itr_end = len(lines)
        for itr in range(cur_itr_start,cur_itr_end):
            
            yield lines[itr][0]
            
        
        
        
        
for i in generateData(file_directory,1,lines):
    print(i)
    
    
    
        
    

