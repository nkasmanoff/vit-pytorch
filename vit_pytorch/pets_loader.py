import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import os
import sys

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def get_pet_data(dset='oxford_pet',
                 data_dir = '../data/',
                 val_size = 0.20,
                   batch_size = 64):

    if dset == 'oxford_pet':
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', data_dir = data_dir, shuffle_files = True, with_info=True)
    
    else:
        print("Must select oxford_pet")
        sys.exit()
    
    train_size = int(info.splits['train'].num_examples)
    val_size = int(train_size * val_size)
    train_size -= val_size
    
    train_ds = dataset['train'].take(train_size).map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = dataset['train'].skip(train_size).map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = dataset['test'].map(load_image_test)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
    
    