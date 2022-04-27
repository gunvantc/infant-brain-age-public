import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import os
from tqdm import tqdm
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import random
import pickle


from data_loader_utils import *
from models import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import multi_gpu_model

from keras import backend as K

def train_model(model, generators, sv_dir, ngpus=2, verbose=2, num_epochs=50, **configs):
    
    training_generator, validation_generator, test_generator = generators
    
    stopping = EarlyStopping(patience=30)
    reduce_lr = ReduceLROnPlateau(
        factor=0.1,
        patience=8,
        min_lr=configs['learning_rate'] * 0.001)
    
    checkpointer = ModelCheckpoint(filepath=os.path.join(sv_dir,
                "{val_loss:.3f}-{epoch:03d}-{loss:.3f}.hdf5"), save_best_only=True)
    
    test_eval = Test_Eval(test_generator)

    opt = Adam(lr=configs['learning_rate'])
    parallel_model = multi_gpu_model(model, gpus=ngpus) if ngpus>1 else model

    # model.compile(loss='mse', optimizer = opt, metrics=['mae'])
    parallel_model.compile(loss='mae', optimizer = opt, metrics=['mae'])
    
    history = parallel_model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True, workers=16,
                        epochs=num_epochs, callbacks=[stopping, reduce_lr, checkpointer, test_eval], verbose=verbose)

    # model.fit_generator(training_generator,
    #                     validation_data=validation_generator,
    #                     epochs=50, callbacks=[stopping, reduce_lr, checkpointer])

    # model.fit([x_train, train_genders], y_train, validation_data = ([x_val, val_genders] y_val), epochs=100)
    
    return history


def main(parser):
   
    #Varaibles to potentially change
    meta_file = 'T1_T2_registered.csv'
    n_epochs = 50

    
    
    model_type, mods, model_name_base, rt_sv_dir, ngpus, scale, batch_size, n_iter, exclude_abnorm, exclude_con, colors_to_inc = parser.model_type, parser.mods, parser.model_name_base, parser.rt_sv_dir, int(parser.ngpus), float(parser.scale), int(parser.batch_size), int(parser.n_iter), bool(parser.exclude_abnorm == 'true'), bool(parser.exclude_con=='true'), parser.colors_to_inc
    learning_rate = float(parser.learn_rate)

    mods = mods.split('|')
    if colors_to_inc != '':
        colors_to_inc = colors_to_inc.split('|')
    else:
        colors_to_inc = None
     
    #Optional step to exclude based on accession numbers
#     exclude_df = pd.read_excel('study_descrip_exclude_list.xlsx')
#     exclude_accs = exclude_df[exclude_df['mod'].isin(mods)].acc.tolist()
    exclude_accs = []
    
    mri_types = [x+'_registered_images' for x in mods] #what fields the modality file paths are called in meta_file

    #Make unique model name for each iteration
    model_name_base = '_'.join(mods) + model_name_base
    model_name = model_name_base % random.randrange(0,999999)
    print(model_name)
    if not os.path.isdir(os.path.join(rt_sv_dir, model_name)):
        os.mkdir(os.path.join(rt_sv_dir, model_name))
   
    print('model_type, mods, npus, scale, batch_size, n_iter, exclude_abnorm, exlcude_con, colors_to_inc')
    print(model_type,mods,ngpus,scale,batch_size,n_iter,exclude_abnorm,exclude_con,colors_to_inc)
    
    n_channels = len(mods)
    mri_shape = (int(172*scale), int(220*scale), int(156*scale))    
    
    histories = []
    for iii in range(n_iter):
        
        #Make unique model name for each iteration
        print('Starting iteration...', iii)
        this_iter_model_name = os.path.join(rt_sv_dir, model_name, str(iii))
        if not os.path.isdir(this_iter_model_name):
            os.mkdir(this_iter_model_name)
        print(this_iter_model_name)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        K.set_session(sess)

        x_train_files, x_val_files, x_test_files, y_train, y_val, y_test, y_train_weights, y_val_weights, y_test_weights, gest_weights = load_data_multiple_v2_kfold(file_types=mri_types, 
                         meta_file=meta_file,
                         exclude_accs=exclude_accs,
                         exclude_abnorm=exclude_abnorm, 
                         exclude_con=exclude_con,
                         colors_to_inc=colors_to_inc,
                         kfold_index=iii,
                         num_folds=n_iter)
        
        #Dump all data splits into pickle
        all_train_data = {'x_train':x_train_files, 'x_val':x_val_files, 'x_test':x_test_files, 'y_train':y_train, 'y_val':y_val, 'y_test':y_test}
        with open(os.path.join(this_iter_model_name, 'all_training_data'), 'wb') as handle:
            pickle.dump(all_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        #Parameters for generators
        train_gen_params = {
            'batch_size':batch_size, 
            'dim':mri_shape,
            'n_channels':n_channels,
            'shuffle':True,
            'resample':scale,
            'augment':True
        }
        val_gen_params = {
            'batch_size':batch_size, 
            'dim':mri_shape,
            'n_channels':n_channels,
            'shuffle':False,
            'resample':scale,
            'augment':False
        }
        test_gen_params = {
            'batch_size':1, 
            'dim':mri_shape,
            'n_channels':n_channels,
            'shuffle':False,
            'resample':scale,
            'augment':False
        }
        custom_train_gen_params = {
            'batch_size':1, 
            'dim':mri_shape,
            'n_channels':n_channels,
            'shuffle':False,
            'resample':scale,
            'augment':False
        }

        print('Loading in train MRIs')
        train_loader = MRI_Loader(x_train_files,y_train,y_train_weights,**train_gen_params)
        print('Loading in val MRIs')
        val_loader = MRI_Loader(x_val_files,y_val,y_val_weights,**val_gen_params)
        print('Loading in test MRIs')
        test_loader = MRI_Loader(x_test_files,y_test,y_test_weights,**test_gen_params)   
        
        # Generators
        training_generator = DataGeneratorPair(train_loader, **train_gen_params)
        validation_generator = DataGeneratorPair(val_loader, **val_gen_params)
        test_generator = DataGeneratorPair(test_loader, **test_gen_params)
        
        generators = (training_generator, validation_generator, test_generator)
        
        
        if model_type == 'uk_biobank':
            network = uk_biobank_network(scale,n_channels)
        elif model_type == 'brain_tumor':
            network = brain_tumor_network(scale,n_channels)
        elif model_type == 'uk_biobank_multichannel':
            network = multichannel_uk_biobank_network(scale,n_channels)
        elif model_type == 'brain_tumor_multichannel':
            network = brain_tumor_multimodal_network(scale,n_channels)
        else:
            print('Invalid model type entry!',model_type)
            return
                 
        model = network.build_model()
        model.summary()
        configs = network.get_configs()
        configs['learning_rate'] = learning_rate
        print(configs)

        #Train model
        history = train_model(model, generators, sv_dir=this_iter_model_name, ngpus=ngpus, num_epochs=n_epochs, **configs)
        histories.append(history)
        
        
        with open(os.path.join(this_iter_model_name,'train_history.p'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description = 'Run ml model')
    parser.add_argument("-t", "--model_type",  help = "Which model to train on", required=True)
    parser.add_argument("-m", "--mods", help = "Modalities to train on, separated by |", required=True)
    parser.add_argument("-n", "--model_name_base", help = "Model base Name", required=True)
    parser.add_argument("-r", "--rt_sv_dir", help = "Root save directory to store model in", required=True)
    parser.add_argument("-g", "--ngpus", help = "Numbers of gpus to use", default=2)
    parser.add_argument("-s", "--scale", help = "How much to up or down sample mris", default=10)
    parser.add_argument("-b", "--batch_size", help = "Batch size for model", default=16)
    parser.add_argument("-i", "--n_iter", help = "Number of iterations to run model", default=5)
    parser.add_argument("-a", "--exclude_abnorm", help = "Boolean to exclude manually determined abnormal cases", default=True)
    parser.add_argument("-c", "--exclude_con", help = "Boolean to exclude manually determined contrast cases", default=True)
    parser.add_argument("-x", "--colors_to_inc", help = "Which quality/qualities (color(s)) of scans to include, separated by |")
    parser.add_argument("-l", "--learn_rate", help = "Learning rate")
    
    args = parser.parse_args()
        
    main(args)
