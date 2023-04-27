import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.metrics import auc, roc_curve

dataset_location = './dataset/train'

#create training and testing directories
train_location = os.path.join(dataset_location,'training')
if not os.path.isdir(train_location):
  os.mkdir(train_location)

dog_train_location = os.path.join(train_location,'dog')
if not os.path.isdir(dog_train_location):
  os.mkdir(dog_train_location)

cat_train_location = os.path.join(train_location,'cat')
if not os.path.isdir(cat_train_location):
  os.mkdir(cat_train_location)

test_location = os.path.join(dataset_location,'testing')
if not os.path.isdir(test_location):
  os.mkdir(test_location)

dog_test_location = os.path.join(test_location,'dog')
if not os.path.isdir(dog_test_location):
  os.mkdir(dog_test_location)

cat_test_location = os.path.join(test_location,'cat')
if not os.path.isdir(cat_test_location):
  os.mkdir(cat_test_location)

#split the data into training and testing datasets for each of the binary classes (dog,cat)
training_dataset_size = 0.80
cat_imgs_size = len(glob.glob('./dataset/train/cat*'))
dog_imgs_size = len(glob.glob('./dataset/train/dog*'))

for i,img in enumerate(glob.glob('./dataset/train/cat*')):
  if i < (cat_imgs_size * training_dataset_size):
    shutil.move(img,cat_train_location)
  else:
    shutil.move(img,cat_test_location)

for i,img in enumerate(glob.glob('./dataset/train/dog*')):
  if i < (dog_imgs_size * training_dataset_size):
    shutil.move(img,dog_train_location)
  else:
    shutil.move(img,dog_test_location)


samples_dog = [os.path.join(dog_train_location,np.random.choice(os.listdir(dog_train_location),1)[0]) for _ in range(8)]
samples_cat = [os.path.join(cat_train_location,np.random.choice(os.listdir(cat_train_location),1)[0]) for _ in range(8)]

nrows = 4
ncols = 4
npics = nrows+ncols

fig, ax = plt.subplots(nrows,ncols,figsize = (10,10))
ax = ax.flatten()

for i in range(nrows*ncols):
  if i < npics:
    image = plt.imread(samples_dog[i%npics])
    ax[i].imshow(image)
    ax[i].set_axis_off()
  else:
    image = plt.imread(samples_cat[i%npics])
    ax[i].imshow(image)
    ax[i].set_axis_off()
plt.show()

print('dogs training data')
dog_training_names = os.listdir(dog_train_location)
print(dog_training_names[:10])
print('cats training data')
cat_training_names = os.listdir(cat_train_location)
print(cat_training_names[:10])
print('dogs testing data')
dog_test_hames = os.listdir(dog_test_location)
print(dog_test_hames[:10])
print('cats testing data')
cat_test_names = os.listdir(cat_test_location)
print(cat_test_names[:10])
print('total training dogs images:', len(os.listdir(dog_train_location)))
print('total training cats images:', len(os.listdir(cat_train_location)))
print('total testing dogs images:', len(os.listdir(dog_test_location)))
print('total testing cats images:', len(os.listdir(cat_test_location)))


# Rescale images
rescale_factor = 1/255
train_image_generator = ImageDataGenerator(rescale=rescale_factor)
test_image_generator = ImageDataGenerator(rescale=rescale_factor)

target_classes = ['dog', 'cat']
#setup image dataflows in batches
training_batch_size = 120
train_data_generator = train_image_generator.flow_from_directory('./dataset/train/training', classes = target_classes, class_mode='binary', target_size=(200, 200), batch_size=training_batch_size)
testing_batch_size = 19
test_data_generator = test_image_generator.flow_from_directory('./dataset/train/testing/', shuffle=False, classes = target_classes, class_mode='binary', target_size=(200, 200), batch_size=testing_batch_size)

binary_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), tf.keras.layers.Dense(128, activation=tf.nn.relu), tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
print('model summary')
print(binary_model.summary())

binary_model.compile(optimizer = tf.optimizers.Adam(),loss = 'binary_crossentropy',metrics=['accuracy'])

epoch_steps = 5
test_steps = 5
epochs = 20
binary_model.fit(train_data_generator,validation_data = test_data_generator, steps_per_epoch=epoch_steps, epochs=epochs, validation_steps=test_steps,verbose=1)
print('binary model evaluation')
print(binary_model.evaluate(test_data_generator))

#test created model
test_data_generator.reset()
predicted_classes = binary_model.predict(test_data_generator,verbose=1)

false_positive_ratio, true_positive_ratio, _ = roc_curve(test_data_generator.classes, predicted_classes)
roc_auc = auc(false_positive_ratio, true_positive_ratio)

#visualize model convergence
plt.figure()
line_width = 2
plt.plot(false_positive_ratio, true_positive_ratio, lw=line_width, color='yellow', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=line_width, color='green', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()
