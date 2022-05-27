import os
# import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

# Directory with angry images
With_Mask = os.path.join(r'E:\ML Project\dataset\with mask')

# Directory with Happy images
Without_Mask_Dir = os.path.join(r'E:\ML Project\dataset\without mask')


train_With_Mask = os.listdir(With_Mask)
print(train_With_Mask[:5])

train_Without_Mask = os.listdir(Without_Mask_Dir)
print(train_Without_Mask[:5])
batch_size=16

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'E:\ML Project\dataset',
                                                    # This is the source directory for training images
                                                    target_size=(200,200),  # All images will be resized to 48 x 48
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',

                                                    # Specify the classes explicitly
                                                    classes=['with mask', 'without mask'],
                                                    # Since we use categorical_crossentropy loss, we need categorical labels
                                                    class_mode='binary')
target_size = (200,200)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 48*48 with 3 bytes color

    # The first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200,200, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The third convolution
    #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 64 neuron in the fully-connected layer
    tf.keras.layers.Dense(64, activation='relu'),

    # 2 output neurons for 2 classes with the softmax activation
    tf.keras.layers.Dense(2, activation='relu')
    ])
model.summary()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])  # RMSprop(lr=0.001),categorical
# Total sample count
total_sample = train_generator.n
# Training
num_epochs=15
model.fit_generator(train_generator, steps_per_epoch=int(total_sample / batch_size),
                    epochs=num_epochs, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open(r"E:\ML Project\model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"E:\ML Project\model.h5")
print("Saved model to disk")