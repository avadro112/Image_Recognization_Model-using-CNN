
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
##tf.__version__ it should be 2.x.x
##datagenrator here to format the data into input(reqquired) datastructer
##
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

##started creating layers using sequence ()
#code below is simple implentation of ANN aglorithms 
##tf.keras.layers.Conv2D is used for filter optimization over the image matrix 
cnn = tf.keras.models.Sequential()


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##actual training 
##below line will take sometime to train the model (it depends on machine) usually cloud implemnetation of CNN/ANN are faster.
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


##here is the image top be predicated method using numpy which converts the image to array input for the algorithm above 
#image is being imported using keras.preprocessing.image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/image_to_be_predicted/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'catagory 1'
else:
    prediction = 'catagory 2'
print(prediction)
