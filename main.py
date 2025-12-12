import os
import cv2
import time
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import graphviz
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from tensorflow.keras.utils import plot_model

def load_df(path):
    x, y = [], []
    
    for folder in tqdm(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):  
            for file in os.listdir(folder_path):
                x.append(os.path.join(folder_path, file))
                y.append(folder)
        
    return pd.DataFrame({'image': x, 'label': y})

train_df = load_df("data/slash-dataset-480p")
val_df   = load_df("data/slash-dataset-1080p")
test_df  = load_df("data/test-images")

print(train_df.head())
root_path = "data/slash-dataset-480p"

slash_df = load_df(root_path)

slash_df['id'] = slash_df['image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

print(slash_df.head())

print(slash_df.shape)
slash_df.head()

X = []
time1 = time.time()

for i, img_path in enumerate(slash_df.image):    
    img = cv2.imread(img_path)[:,:,::-1]
    X.append(img)
    
time2 = time.time()
time3 = np.round(time2 - time1)
print(time3, "sec")

selected_columns = ['image', 'label']
label_counts = slash_df[selected_columns].groupby('label').count()
label_counts.plot(kind='bar', figsize=(8, 6), color='red')
plt.title('Count of Retail Products by Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

SHAPE = X[0].shape[0]
BATCH_SIZE = 8
SHUFFLE_SIZE = 1_000
def get_category(y_hat, cols):
    max_value = y_hat.max()
    for index, category in enumerate(y_hat):
        if category == max_value:
            return cols[index]
    
    return 'unknown'

def augment_image(image, label):
    img = tf.image.random_flip_left_right(image)
    img = tf.image.random_flip_up_down(img)
    
    img = tf.image.random_brightness(img, max_delta=0.2)
    
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    
    return img, label

def tensor_slices_dataset(x, y, shuffle_size = SHUFFLE_SIZE, batch_size = BATCH_SIZE, training=False):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    
    if training:
        augmented_data = data.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.concatenate(augmented_data)

        
    data = data.shuffle(shuffle_size)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    
    return data

if X[0].dtype == 'uint8':
    for index in tqdm(range(len(X))):
        normalized_image = X[index] / 255.0
        rounded_image = np.round(normalized_image, decimals=5)

        X[index] = rounded_image

        y = pd.get_dummies(slash_df['label'])
_columns = y.columns 

print(_columns)
y.head()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=slash_df['label'])
print(y_train.shape, y_val.shape)

idx = random.randint(0, y_train.shape[0] - 1)

print(y_train.iloc[idx])
plt.imshow(X_train[idx])

train_tensor = tensor_slices_dataset(X_train, y_train, training=True)
val_tensor = tensor_slices_dataset(X_val, y_val)

train_tensor.take(1) 

for images, labels in train_tensor.take(1):
    img = images[0].numpy()   
    plt.imshow(img)
    plt.title(str(labels[0].numpy()))
    plt.axis("off")
    plt.show()

    for i in range(8):                
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i].numpy()) 
        plt.xlabel(get_category(labels[i].numpy(), _columns))
        plt.xticks([])
        plt.yticks([])

plt.show()


best_model_filepath = "d:/kothai/finalyearProject/product_classification/model/model_checkpoint.keras"

os.makedirs(os.path.dirname(best_model_filepath), exist_ok=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(SHAPE, SHAPE, 3))
base_model.trainable = False


input_layer = tf.keras.layers.Input(shape=(SHAPE, SHAPE, 3))

tl_model = base_model(input_layer, training=False)

x = tf.keras.layers.Flatten()(tl_model)

x = tf.keras.layers.Dense(units=256, activation='leaky_relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(units=128, activation='leaky_relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

output_layer = tf.keras.layers.Dense(units=_columns.shape[0], activation='softmax')(x)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

model.fit(train_tensor,
          batch_size=BATCH_SIZE,
          epochs=30,
          validation_data=val_tensor,
          callbacks = [model_checkpoint_callback])


model = load_model('/kaggle/working/model/model_checkpoint.keras')

test_loss, test_accuracy = model.evaluate(val_tensor)
print('Test loss: {0:.4f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


