# Infant Brain Age Prediction

This repository is an accompaniment to the research article titled, "Using Deep Learning to Predict Neonatal and Infant Brain Age from Myelination on Brain MRI". This repository contains image preprocessing and code to train regression model for predicting infant brain age from MRIs of one or more modalities using a 3D CNN. During training, this code divides data into K-folds to allow for testing of each sample exactly once.

## Requirements
- Image directory structure of `$ROOT_DIR/$ACCESSION_NUMBER/$DICOM_OR_NIFTI_FILES`
- Valid conda install
- Skullstripping library, e.g. [Deepbrain](https://pypi.org/project/deepbrain/)
- FSL (for registration with [FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT))

## Installation
### Code
```
cd /directory/where/you/want/the/code
git clone git@github.com:gunvantc/infant-brain-age-public.git
```
### Python Environment
A valid conda install is needed (Anaconda, Miniconda).
Install the environment from the brain_env.yml file.

```
# Standard install:
conda env create -f brain_env.yml
# Specify where to install:
conda env create --prefix /path/to/install -f brain_env.yml
```

## Preprocessing

### Convert DICOM to Nifti
- Use `preproc_pipeline.ipynb`
- Requires a DataFrame variable (`grouped_master_file_df`) with dicom folder names for each modality in separate fields and the rt_dir in 'bases'
- Example DataFrame Header: `'T1'  | 'T2'  |  'bases'`, such that `os.path.join(row.bases,row.T1)` yields DICOM folder to process. The field `bases` should follow the pattern of `$root_dir/$accesion_number/'`.

### Skullstrip
- Modify `preproc_pipeline.ipynb` to use the skullstripping library of your choice. An internal skullstripping model was used for the associated publication that we are not able to release.

### Register to Atlas
- Use `register.ipynb`. This method uses a glob matching pattern to find eligible files to register (i.e. files that are of the correct modality and are skullstripped)
- Change input and output directories as necessary.


## Train Regression Model
### Edit `start_training.sh`:

| Variable    | Description | Example Value(s) | 
| ----------- | ----------- | ----------- |
| model_type      | Model structure to use. See `models.py`  | `'uk_biobank'` or `'brain_tumor'` or  `'uk_biobank_multichannel'` or `'brain_tumor_multichannel'`  | 
| mods   | Modalities to use, separated by `'\|'`        | `'T1\|T2'` |
|model_name_base |Format for naming the model folder | `'_registered_biobank_regress_%06d'` |
|rt_sv_dir | Directory in which to save model folder | `'/data/rauschecker1/infantBrainAge/models'` |
|ngpus | Number of GPUs to use in training | `2` |
|scale | Factor by which to scale images. Scale down to reduce GPU use. | `0.5` |
| batch_size | Training batch size | `16`|
|learn_rate| Training learning rate | `1e-3` |
| n_iter | Number of kfolds to run model on | `10` |
|colors_to_inc| What quality of images to include based on prior classification separated by `'\|'`. Value can be `''` if you want to include all images. |`'GREEN\|YELLOW\|ORANGE\|RED'`|
|exclude_abnorm|Whether to exclude images with diagnoses that have been labeled as such|`false`|
|exclude_con|Whether to exclude post-contrast images that have been labeled as such|`false`|

### Edit additional configs in `network_regress_model.py`:
| Variable    | Description | Example Value |
| ----------- | ----------- | ----------- |
| meta_file      | Pandas csv or excel file with integer field of acc_nums (Accesion numbers) and fields for modality-specific image files in `'$MODALITY_registered_images'` format. | `'T1_T2_registered.csv'`  |
| n_epochs   | Number of epoch to run training  | `50` |

### Start training:
```
chmod +x start_training.sh
start_training.sh
```
Performance during training can be monitored in the console. Results on training, validation, and test sets will output every epoch. Data used during each kfold iteration, the best model for each iteration, and keras's training history will save in `$MODEL_DIR/$KFOLD_ITER/`.

*Written by Gunvant Chaudhari, 2020-21.*