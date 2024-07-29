import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from tensorflow.keras import regularizers

imgsz = (256,256)
batsz = 32
train_data = image_dataset_from_directory("./dataset/train",
                                        labels="inferred",
                                        label_mode='int',
                                        seed = 1337,
                                        image_size= imgsz,
                                        batch_size = batsz)
test_data = image_dataset_from_directory("./dataset/test",
                                        labels="inferred",
                                        label_mode='int',
                                        seed = 1337,
                                        image_size= imgsz,
                                        batch_size = batsz)

def normalize(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_data = train_data.map(normalize)
test_data = test_data.map(normalize)

model1 = Sequential()

model1.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model1.add(Conv2D(64,kernel_size=(4,4),padding='valid',activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model1.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model1.add(Flatten())

model1.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.2))

model1.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.2))

model1.add(Dense(1, activation='sigmoid'))


model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model1.summary()
model1.fit(train_data,epochs=15,validation_data=test_data)

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(4, 4), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model2.add(Conv2D(64, kernel_size=(4, 4), padding='valid', activation='relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size=(4,4), strides=2, padding='valid'))


model2.add(Flatten())

model2.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.2))

model2.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model2.add(Dropout(0.2))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model2.summary()

model2.fit(train_data,epochs=15,validation_data=test_data)

model3 = Sequential()


model3.add(Conv2D(32, kernel_size=(4, 4), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model3.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model3.add(Conv2D(128, kernel_size=(4, 4), padding='valid', activation='relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))


model3.add(Flatten())

model3.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.2))

model3.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model3.add(Dropout(0.2))

model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model3.summary()
model3.fit(train_data,epochs=15,validation_data=test_data)

from keras.models import Model
from keras.layers import Average

input_layer_1 = model1.input
input_layer_2 = model2.input
input_layer_3 = model3.input

output1 = model1.output
output2 = model2.output
output3 = model3.output

average_layer = Average()([output1, output2,output3])

ensemble_model = Model(inputs=[input_layer_1,input_layer_2,input_layer_3], outputs=average_layer)

ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
