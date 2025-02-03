# Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2  # Pre-trained model for transfer learning
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import numpy as np
from PIL import Image

# Configuration
DATASET_DIR = 'all_dataset'  # Directory containing plant images organized in class folders
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# Create directories for organizing the processed data
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/validation', exist_ok=True)

# Automatically detect all plant species classes from the dataset directory
class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
print(f"Found {len(class_names)} classes")

# Define the plant species classification model
def create_model(num_classes):
    """
    Creates a transfer learning model based on MobileNetV2 for plant classification
    Args:
        num_classes: Number of plant species to classify
    Returns:
        A compiled Keras model
    """
    # Load pre-trained MobileNetV2 model without top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    # Add global average pooling to reduce parameters
    x = GlobalAveragePooling2D()(x)
    # Add a dense layer for feature learning
    x = Dense(1024, activation='relu')(x)
    # Final classification layer
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Set up data augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift image horizontally
    height_shift_range=0.2,  # Randomly shift image vertically
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of data for validation
)

# Create data generators for training and validation
print("Setting up data generators...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=32,  # Process 32 images per batch
    class_mode='categorical',  # Multi-class classification
    subset='training'  # Use training split
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use validation split
)

# Initialize and compile the model
print("Creating and compiling model...")
num_classes = len(class_names)
model = create_model(num_classes)
model.compile(
    optimizer='adam',  # Adam optimizer for adaptive learning rate
    loss='categorical_crossentropy',  # Standard loss for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Set up model checkpoint to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'crop (1).h5',  # Save model to this file
    save_best_only=True,  # Only save when model improves
    save_weights_only=False,  # Save entire model, not just weights
    monitor='val_accuracy',  # Monitor validation accuracy
    mode='max'  # Save when accuracy increases
)

print("Starting model training...")
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=3,  # Number of complete passes through the dataset
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkpoint],
    verbose=1  # Show progress bar
)

print("Training completed. Saving weights...")
model.save_weights('crop_weights (1).h5')

print("Creating segmentation model...")
# Define the U-Net model for leaf segmentation
def create_segment_model():
    """
    Creates a simplified U-Net model for leaf segmentation
    Returns:
        A compiled Keras model for image segmentation
    """
    inputs = tf.keras.Input(shape=(256, 256, 3))
    
    # Encoder path: compress the image and extract features
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge: lowest resolution processing
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    
    # Decoder path: recover spatial resolution
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = tf.keras.layers.concatenate([up1, conv2])  # Skip connection
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
    
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = tf.keras.layers.concatenate([up2, conv1])  # Skip connection
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
    
    # Final layer for pixel-wise classification
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

print("Training segmentation model...")
segment_model = create_segment_model()
segment_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary classification per pixel
    metrics=['accuracy']
)

# Train the segmentation model
print("Training segmentation model...")
segment_history = segment_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the segmentation model
print("Saving segmentation model...")
segment_model.save('leaf.h5')
segment_model.save_weights('leaf_weights.h5')

print("Training complete! Both models have been saved.")
