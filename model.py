import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import ResNet50, ResNet101, ResNet50V2, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.python.ops.gen_math_ops import mod_eager_fallback
import os

def create_model(input_shape, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    new_model = base_model.output
    new_model = GlobalAveragePooling2D()(new_model)
    new_model = Flatten()(new_model)
    
    new_model = Dense(2048, activation='relu')(new_model)
    new_model = Dropout(0.2)(new_model)
    output_layer = Dense(num_classes, activation='softmax')(new_model)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    optimizer = Adam(lr= 1e-3)
      
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])


    
    return model


def create_model_ResNet101(input_shape, num_classes):
    base_model = ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    new_model = base_model.output
    new_model = GlobalAveragePooling2D()(new_model)
    new_model = Flatten()(new_model)
    
    new_model = Dense(2048, activation='relu')(new_model)
    new_model = Dropout(0.2)(new_model)
    output_layer = Dense(num_classes, activation='softmax')(new_model)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    optimizer = Adam(lr= 1e-3)
      
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])


    return model


def create_model_ResNet50V2(input_shape, num_classes):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    new_model = base_model.output
    new_model = GlobalAveragePooling2D()(new_model)
    new_model = Flatten()(new_model)
    
    new_model = Dense(2048, activation='relu')(new_model)
    new_model = Dropout(0.2)(new_model)
    output_layer = Dense(num_classes, activation='softmax')(new_model)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    optimizer = Adam(lr= 1e-3)
      
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])


    return model

def create_model_ResNet101V2(input_shape, num_classes):
    base_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    new_model = base_model.output
    new_model = GlobalAveragePooling2D()(new_model)
    new_model = Flatten()(new_model)
    
    new_model = Dense(2048, activation='relu')(new_model)
    new_model = Dropout(0.2)(new_model)
    output_layer = Dense(num_classes, activation='softmax')(new_model)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    optimizer = Adam(lr= 1e-3)
      
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])


    return model


def create_model_ResNet152(input_shape, num_classes):
    base_model = ResNet152(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    new_model = base_model.output
    new_model = GlobalAveragePooling2D()(new_model)
    new_model = Flatten()(new_model)
    
    new_model = Dense(2048, activation='relu')(new_model)
    new_model = Dropout(0.2)(new_model)
    output_layer = Dense(num_classes, activation='softmax')(new_model)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    optimizer = Adam(lr= 1e-3)
      
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])


    return model

def create_model_ResNet152V2(input_shape, num_classes):
    base_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True
    new_model = base_model.output
    new_model = GlobalAveragePooling2D()(new_model)
    new_model = Flatten()(new_model)
    
    new_model = Dense(2048, activation='relu')(new_model)
    new_model = Dropout(0.2)(new_model)
    output_layer = Dense(num_classes, activation='softmax')(new_model)
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    optimizer = Adam(lr= 1e-3)
      
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])


    return model    


def create_callbacks(save_path):
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path,'Satellite_finetuned_{epoch:02d}.h5'),
                        monitor='val_accuracy',
                        save_best_only=True,
                        verbose=1, period =10)

    early_stop = EarlyStopping(monitor='val_accuracy',
                            patience=10,
                            restore_best_weights=True,
                            mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                patience=3, min_lr=0.00001)

    return [checkpoint, early_stop, reduce_lr]                            



