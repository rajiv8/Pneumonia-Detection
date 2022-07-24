# importing the required libraries 
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation

# function to load model
def LoadCnn():
    model=Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(120, 120, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('Pnemonia_prediction_cnn_model.h5')
    return model



def Detect(imagePath):
    image=cv2.imread(imagePath)
    # grayImage=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    resized=cv2.resize(image,(120,120))
    normalized=resized/255
    reshaped=np.reshape(normalized,(1,120,120,3))
    result=model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    prob=np.max(result,axis=1)[0]
    prob=round(prob,2)*100
    image=cv2.resize(image,(960,640))
    cv2.putText(image,str(category[label]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2)
    cv2.putText(image,str(round(prob,2)),(100,200),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2)
    cv2.imshow('Detect',image)
    k = cv2.waitKey(0) & 0xFF
    if k==27:
        cv2.destroyAllWindows()
    print(result)
    print(label)
    print(prob)


if __name__=="__main__":
    model=LoadCnn()
    category={0:'Normal',1:'Pnemonia'}
    
    # path to the image which needs to be classified
    Detect("D:/Pneumonia detection/chest_xray/chest_xray/test/NORMAL/IM-0003-0001.jpeg")
