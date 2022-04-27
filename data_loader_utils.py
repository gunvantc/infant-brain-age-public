import numpy as np
import pandas as pd
import nibabel as nib
import glob
import os
from tqdm import tqdm
import keras
from sklearn.model_selection import train_test_split
from scipy import ndimage
from sklearn.utils.class_weight import compute_class_weight
import random
import pdb
from multiprocessing import Pool
from itertools import repeat

def compile_subset(subset_dir='/data/rauschecker1/infantBrainAge/resources/subanalyses', 
                   subset=[]):
    '''
        Compile cases that have diagnoses in the radiology report, as previouly assembled in subset_dir
        as excel files
    '''
    compiled_set = pd.DataFrame()
    subset_files = glob.glob(os.path.join(subset_dir, '*.xlsx'))
    for each in subset_files:
        if each not in subset:
            compiled_set = compiled_set.append(pd.read_excel(each))
    
    return compiled_set


def load_data_multiple_v2_kfold_clean_data(labels_path='/data/rauschecker1/infantBrainAge/resources/li sort repaired v2_gc.csv',                            meta_file='T1_T2_ADC_aligned_resampled_list.csv', con_file=None, exclude_con=False,
                   file_types=['T1_skullstripped_resampled'],
                      exclude_accs=[], exclude_abnorm=True,  min_res=0, colors_to_inc=None,
                      rand_seed=2021, kfold_index=-1, num_folds=10, weighting='consecutive'):
    '''
       Create training splits and weighting using linear weighting, now with kfold support.             
                                                                                        
       param labels_path: path of the labels file, a pandas csv file with fields 'Accession Number' and 
            'Corrected GA at MRI (weeks)'
       param meta_file: path of pandas csv or excel file with images to use in model, with integer field 
            'acc_nums'/'accession_num' and images in modality-specific fields (e.g.'T1_registered_images')
       param con_file: path of pandas csv or excel file for exlcusion of post-contrast images. Should
             contain integer field of 'acc_nums' and boolean fields $MODALITY_con (e.g. 'T1_con') that
             signify whether the corresponding image has contrast
       param file_types: list of what fields in meta_file contain image paths to use
             (e.g. ['T1_registered_images','T2_registered_images'])
       param exclude_accs: list of accession numbers to exclude
       param exclude_abnorm: boolean of whether to exclude images with diagnoses e.g. hydrocephalus that
             have been previously assembled
       param min_res: minimum resolution (in voxels) of orignal nifti file (before preprocessing) to be 
             included in the model. If 0, the orignal nifti file is not assessed for resolution, so it
             improves loading speed
       param colors_to_inc: optional list of what quality scans to include, e.g. ['GREEN','RED']
       param rand_seed: pseudorandom generator seed to allow for same division in multiple runs of this
             method
       param kfold_index: index of the kfold is currently being analyzed, in range 0 to num_folds-1
       param num_folds: number of k-folds to use in divding the dataset
       param weighting: type of sample weighting to use. Options are:
                    'consecutive': linear weight based on bins of size (age_max-age_min)/10. The weight 
                             value starts at one at bin0 and increase by one, up to 10 at bin9.
                    'percentile': weight based on bins of size as above, but with weight value being
                             the inverse of number of images in the bin.
                    'none': weight of 1 for every sample.
    ''''    
    
    
    
    print('Doing %d fold-wise analysis, on fold %d' % (num_folds, kfold_index))
    
    if weighting not in ['consecutive','percentile','none']:
        print('unrecognized weighting')
        return None
    
    if meta_file[-3:]=='csv':
        master_files = pd.read_csv(meta_file)
    else:
        master_files = pd.read_excel(meta_file)
    
    master_files = master_files.fillna('No Path')
    exclude_accs = [str(x).strip() for x in exclude_accs]
    if 'acc_nums' in master_files.columns:
        master_files = master_files.rename(columns={'acc_nums':'accession_num'})
    
    if colors_to_inc and 'study_quality' in master_files.columns:
        print('including only: ', colors_to_inc)
        master_files = master_files[master_files.study_quality.isin(colors_to_inc)]
    else:
        print('Not excluding based on colors')
    
    if con_file:
        if con_file[-3:]=='csv':
            con_files = pd.read_csv(con_file)
        else:
            con_files = pd.read_excel(con_file)
    
    #Check if file types are in meta_file
    for each_type in file_types:
        if each_type not in master_files.columns:
            print('%s not found in file %s!' % (each_type, meta_file))
            return -1
    
    #Compile abnormal scans
    all_abnormal_df = compile_subset().drop_duplicates(subset=['Accession Number'])
    all_abnormal_list = [] if not exclude_abnorm else all_abnormal_df['Accession Number'].apply(str).tolist()
    
    
    #Compile file paths
    proc_files = []
    excluded_manual = 0
    excluded_path = 0
    excluded_res = 0
    excluded_con = 0
    
    for _,row in tqdm(master_files.iterrows()):
        if str(row.accession_num).strip() in exclude_accs or str(row.accession_num).strip() in all_abnormal_list:
            excluded_manual += 1
            continue
        
        these_paths = [row[xx].replace('/export','') for xx in file_types]
#         these_paths = [row[xx].replace('_fullsampled','') for xx in file_types] # Only skullstripped
         
        invalid=False
        for each_path in these_paths:
            
            if not os.path.exists(each_path): 
                invalid=True
                excluded_path += 1
                continue
            
            raw_path = each_path.replace('_skullstripped','').replace('_alignedT1','').replace('_fullsampled','').replace('_registered','')
            if min_res>0:
                if min(nib.load(raw_path).shape) < min_res:
                    invalid=True
                    excluded_res += 1
                    continue
            
            if exclude_con:
                if each_path.split('/')[-1][:2]=='T1' and con_files[con_files.acc_nums == str(row.accession_num)].iloc[0].T1_con:
                    invalid=True
                    excluded_con += 1
                    continue
                elif each_path.split('/')[-1][:2]=='T2' and con_files[con_files.acc_nums == str(row.accession_num)].iloc[0].T2_con:
                    invalid=True
                    excluded_con += 1
                    continue
        
        if not invalid:
            proc_files.append(these_paths)  
    
    
    #Shuffling files with a fixed seed
    random.Random(2021).shuffle(proc_files)
    
    #Compile corresponding gestational ages 
    gest_ages = []
    
    master_labels = pd.read_csv(os.path.join(rt_dir, labels_path))
    for in_f in tqdm(proc_files):
        this_acc = int(in_f[0].split('/')[-2].split('_')[0])
        gest_ages.append(master_labels[master_labels['Accession Number'] == this_acc].iloc[0]['Corrected GA at MRI (weeks)'])
    
#     #manually exclude based on gest ages
#     proc_files = [x for i,x in enumerate(proc_files) if gest_ages[i]]
#     gest_ages = [x for i,x in enumerate(gest_ages) if x]
    
    print(len(proc_files), len(gest_ages))
    print('Excluded: manual %d , path %d , res %d , con %d' % (excluded_manual, excluded_path,
                                                               excluded_res, excluded_con) )
    
    gest_bins = np.linspace(min(gest_ages),max(gest_ages),10,endpoint=False)
    print('Bins:', gest_bins)
    gest_sorted = np.digitize(gest_ages, gest_bins)
    gest_class_weights = compute_class_weight('balanced',classes=np.unique(gest_sorted), y=gest_sorted)
    print('Weights:', gest_class_weights)
    
    
    if weighting == 'consecutive':
        gest_weights = gest_sorted
    elif weighting == 'percentile':
        gest_weights = np.array([gest_class_weights[x-1] for x in gest_sorted])
    else:
        gest_weights = np.array([1 for x in gest_sorted])
    
    print('weighting by', weighting)
    test_indexes = np.arange(int(kfold_index * len(proc_files)/num_folds), int( (kfold_index+1) * len(proc_files)/num_folds)).tolist()
    not_test_indexes = [x for x in range(len(proc_files)) if x not in test_indexes]
#     print(test_indexes)
#     print(not_test_indexes)
    
    proc_files = np.array(proc_files)
    gest_ages = np.array(gest_ages)
    gest_weights = np.array(gest_weights)
    
    x_test = proc_files[test_indexes]
    x_not_test = proc_files[not_test_indexes]
    y_test = gest_ages[test_indexes]
    y_not_test = gest_ages[not_test_indexes]
    y_test_weights = gest_weights[test_indexes]
    y_not_test_weights = gest_weights[not_test_indexes]
    
    #x_train, x_not_train, y_train, y_not_train, y_train_weights, y_not_train_weights = train_test_split(proc_files, gest_ages, gest_sorted, test_size=0.3, random_state=rand_seed, stratify=gest_sorted)
    
    x_val, x_train, y_val, y_train, y_val_weights, y_train_weights = train_test_split(x_not_test, y_not_test, y_not_test_weights, test_size=0.7777777, random_state=rand_seed, stratify=y_not_test_weights)

    x_train=np.array(x_train)
    x_val=np.array(x_val)
    x_test=np.array(x_test)
    
    y_train=np.array(y_train)
    y_val=np.array(y_val)
    y_test=np.array(y_test)
    
    print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)  
    return x_train, x_val, x_test, y_train, y_val, y_test, y_train_weights, y_val_weights, y_test_weights, gest_weights


def crop_to_dim(im, dim):
    'Crops image to dim if necessary'

    if dim==im.shape:
        return im

    x,y,z = dim 
    tx, ty, tz = im.shape
    mx, my, mz = min([x,tx]), min([y,ty]), min([z,tz])
    out_im = np.zeros((x,y,z), dtype=np.float64)

    out_im[ int(x/2-mx/2):int(x/2+mx/2), int(y/2-my/2):int(y/2+my/2),
          int(z/2-mz/2):int(z/2+mz/2)] = \
        im[ int(tx/2-mx/2):int(tx/2+mx/2), int(ty/2-my/2):int(ty/2+my/2),
          int(tz/2-mz/2):int(tz/2+mz/2)]

    return out_im


def normalize(im):
    return (im-np.mean(im))/np.std(im)

def resample_im(im, resample):
    if resample != 1:
        im = ndimage.zoom(im, resample, order = 1)
    return im


def parallel_loader(ID, age, weight, dim, resample):
    ims = tuple()
    for this_ID in ID:
        im = nib.load(this_ID).get_fdata()
        im = resample_im(im, resample)
        im = crop_to_dim(im, dim)
        im = normalize(im)
        im = np.expand_dims(im, 3)
        ims = ims + (im,)

    return np.concatenate(ims, axis=3), age, weight, ID

class MRI_Loader():
    'Loads MRI files into memory'
    
    def __init__(self, mri_list, ages, weights, resample=1, dim=(32,32,32), n_channels=1, num_threads=10, **kwargs):
        self.resample = resample
        self.dim = dim
        self.n_channels = n_channels
        self.weights = weights
        self.load_into_memory(mri_list, ages, weights, num_threads)
        
    def load_into_memory(self, mri_list, ages, weights, num_threads=10):        
        iterable = zip(mri_list, ages, weights, repeat(self.dim), repeat(self.resample))
        with Pool(num_threads) as pool:
            self.all_data = pool.starmap(parallel_loader, iterable)
    
    def __len__(self):
        return len(self.all_data)
                          
    def __getitem__(self, index):
        return self.all_data[index]

def norm_to_ga(in_list, y_trans):
    out_list = []
    for each in in_list:
        out_list.append(each * (y_trans['max']-y_trans['min'])+y_trans['min'])
    return out_list

def transform_labels(y):
    y=np.array(y)
    return 1/(1+np.exp(-0.03*y))

def anti_transform_labels(y):
    y=np.array(y)
    y=np.clip(y, 0.000001, 0.99999)
    return np.log((1-y)/y)/-0.03



class DataGeneratorPair(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, mri_loader, batch_size=32, dim=(32,32,32), n_channels=1,
                 shuffle=True, augment=False, **kwargs):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.mri_loader = mri_loader
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = False
        self.on_epoch_end()      

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.mri_loader) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y, batch_weights = self.__data_generation(indexes)
        
        return X, Y, batch_weights
       
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.mri_loader))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)            

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        X =             np.empty((self.batch_size, *self.dim, self.n_channels))
        Y =             np.empty((self.batch_size), dtype=np.float)
        batch_weights = np.empty((self.batch_size), dtype=np.float)
        
        # Generate data
        for i, ID in enumerate(indexes):
            im, Y[i], batch_weights[i], _ = self.mri_loader[ID]
            X[i,] = self.augmentation(im)
            
        return X, Y, batch_weights
    
    def get_Y(self):
        Y = np.array([data_tuple[1] for data_tuple in self.mri_loader])
        return Y
    
    def get_Weights(self):
        IDS = np.array([data_tuple[2] for data_tuple in self.mri_loader])
        return IDS
    
    def get_IDs(self):
        IDS = np.array([data_tuple[3] for data_tuple in self.mri_loader])
        return IDS
    
    def augmentation(self, im):
        'Augment samples using tranlsation, rotation, flipping, intensity shifts, and gaussian noise'
        if self.augment:
            #Randomly shift 0-3 pixels in each axis
            random_shift=[random.randint(-3,3) for _ in range(3)]
            im = ndimage.shift(im, shift=random_shift)

            #rotate by a small angle
            axes_to_rotate = np.random.choice(3, size=(2,), replace=False)
            angle_to_rotate = random.uniform(-10,10)
            ndimage.rotate(im, angle=angle_to_rotate, axes=axes_to_rotate)

            #Randomly flip along mid-sagittal plane 50% of time
            if random.random()>0.5:
                im = np.flip(im, axis=0)

#             #Histogram shifting (not really)
#             rand_scale = random.uniform(0.9,1.1)
#             im = im*rand_scale
#             std_im_den = np.std(im)
#             rand_shift = random.uniform(-std_im_den/10, std_im_den/10) #shift intensity up/down by 10% of st dev
#             im = im+rand_shift

#             #add some gaussian noise, max 10% std of intensity
#             gaussian_std = np.std(im)/10
#             im = np.add(im, np.random.normal(0, gaussian_std, im.shape))
        
        return im
            

class GPU_Eval(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch==1:
            os.system('nvidia-smi')
            
            
class Test_Eval(keras.callbacks.Callback):
    def __init__(self, test_data_loader):
        super(keras.callbacks.Callback, self).__init__()
        self.test_data_loader = test_data_loader
        
    def on_epoch_end(self, epoch, logs=None):
        print('Test Set Loss and MAE:', self.model.evaluate_generator(self.test_data_loader))