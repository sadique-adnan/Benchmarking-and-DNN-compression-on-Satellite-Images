import os, random, shutil, argparse


def create_test_split(labels, test_split, data_path):
    '''
    Splits the EuroSat into 80% training and 20% testing data
    '''
    random.seed(70)
    for l in labels:
        file_list = os.listdir(os.path.join(data_path, 'train', l))
        random.shuffle(file_list)
        split = int(test_split*len(file_list))
        test_files = file_list[:split]
        if not os.path.exists(os.path.join(data_path, 'test', l)):
            os.makedirs(os.path.join(data_path, 'test', l))
        for file in test_files:
            shutil.move(os.path.join(data_path, 'train', l,file), os.path.join(data_path, 'test', l,file))


                           
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Satellite_Image_Split')
    parser.add_argument('--data_path', type=str, default='/home/sadique/dataset/EuroSAT_tr/', help='data_path')
    parser.add_argument('--test_split', type=float, default=0.2, help='split ratio')
    args = parser.parse_args()

    labels = os.listdir(args.data_path+'train')
    create_test_split(labels, args.test_split, args.data_path)
    
    
    
       