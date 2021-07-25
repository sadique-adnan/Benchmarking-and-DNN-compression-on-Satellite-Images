from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime, os
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score

def create_test_generator(test_dir, batch_size):
    test_gen = ImageDataGenerator(
        rescale=1./255)

    test_generator = test_gen.flow_from_directory(
            directory=test_dir,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode=None,
            color_mode='rgb',
            shuffle=False,
            seed=60
            
        )
    
    return test_generator

def run_test(result_path, saved_model_name, test_generator, BATCH_SIZE, class_labels):
    model = load_model(saved_model_name)
    N_STEPS = test_generator.samples//BATCH_SIZE
    predictions = model.predict_generator(test_generator, steps=N_STEPS)
    
    predicted_classes = np.argmax(np.rint(predictions), axis=1)
    true_classes = test_generator.classes
    acc = calc_metrics(result_path, true_classes, predicted_classes, class_labels)
    return acc


def calc_metrics(result_path, true_classes, predicted_classes, class_labels,  model_name = 'Resnet_152v2'):
    mydir = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except:
        print('Create correct folder structure')

    accuracy = accuracy_score(predicted_classes, true_classes)
    classification_repo = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    print(classification_repo)
    file_name = os.path.join(mydir, 'model_name_%s.csv'%(model_name))
    df = pd.DataFrame(classification_repo)
    df.to_csv(file_name)
    return accuracy

    
    
if __name__ == '__main__':

    TEST_DIR = '../../EuroSAT/test/'
    BATCH_SIZE = 1
   
    test_gen = create_test_generator(TEST_DIR, BATCH_SIZE)
    class_indices = test_gen.class_indices
    class_indices = dict((v,k) for k,v in class_indices.items())

    saved_model_name = 'train_history/resnet_152v2/_satellite_finetuned.h5'
    result_path = 'results/'
    acc = run_test(result_path, saved_model_name, test_gen, BATCH_SIZE, class_indices.values())
    print('Accuracy', acc)
