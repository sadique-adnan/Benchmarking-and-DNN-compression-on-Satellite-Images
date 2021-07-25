from genericpath import exists
import os, argparse
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import create_model, create_model_ResNet101, create_model_ResNet50V2, create_model_ResNet101V2 , create_model_ResNet152, create_model_ResNet152V2, create_callbacks

from utils import restrict_gpu, plot_training
import config as cfg

def create_image_generators(train_dir, batch_size, class_mode):
    train_gen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=60,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    shear_range=0.3,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    validation_split=0.2
                )

    train_generator = train_gen.flow_from_directory(
                        directory=train_dir,
                        target_size=(64, 64),
                        batch_size=batch_size,
                        class_mode=class_mode,
                        subset='training',
                        color_mode='rgb',
                        shuffle=True,
                        seed=69
                    )

    valid_generator = train_gen.flow_from_directory(
                        directory=train_dir,
                        target_size=(64, 64),
                        batch_size=batch_size,
                        class_mode=class_mode,
                        subset='validation',    
                        color_mode='rgb',
                        shuffle=True,
                        seed=69
                    )


    return train_generator, valid_generator    

def main(model_name):
 
    restrict_gpu()

    labels = os.listdir(os.path.join('../../EuroSAT', 'train'))

    TRAIN_DIR = cfg.config['train_dir']
    BATCH_SIZE = cfg.config['BATCH_SIZE']
    NUM_CLASSES = cfg.config['NUM_CLASSES']
    INPUT_SHAPE = cfg.config['INPUT_SHAPE']
    CLASS_MODE = cfg.config['CLASS_MODE']
    result_path = cfg.config['result_path']

    train_generator, valid_generator = create_image_generators(TRAIN_DIR, BATCH_SIZE, CLASS_MODE)
    N_STEPS = train_generator.samples//BATCH_SIZE
    N_VAL_STEPS = valid_generator.samples//BATCH_SIZE
    model_name = model_name
    if model_name == 'resnet_50':
        model = create_model(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    elif model_name == 'resnet_101':
        model = create_model_ResNet101(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    elif model_name == 'resnet_50v2':
        model = create_model_ResNet50V2(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    elif model_name == 'resnet_101v2':
        model = create_model_ResNet101V2(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    elif model_name == 'resnet_152':
        model = create_model_ResNet152(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    elif model_name == 'resnet_152v2':
        model = create_model_ResNet152V2(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    else:
        print('Enter correct model name')    
        
            

    print(model.summary())
    save_path = os.path.join(result_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    callbacks_list = create_callbacks(save_path)
    model_history = model.fit_generator(train_generator,
                             steps_per_epoch=N_STEPS,
                             epochs=50,
                             callbacks=callbacks_list,
                             validation_data=valid_generator,
                             validation_steps=N_VAL_STEPS)

    
    model.save(os.path.join(save_path,'_satellite_finetuned.h5'))
    plot_training(save_path, model_history) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model_Name')
    parser.add_argument('--model', type=str, default='resnet_50', help='choose model name resnet_50  resnet_50v2 resnet_101 resnet_101v2  resnet_152 resnet_152v2')
    args = parser.parse_args()
    main(args.model)

    







