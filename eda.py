import matplotlib.pyplot as plt
import os
import pydicom as dicom
import matplotlib.patches as patches
import pandas as pd
import pydicom

import utils


train_path = '/mnt/rsna/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')

# Generate image paths for train and test data
train_image_paths = utils.generate_image_paths(train_desc, f'{train_path}train_images')
test_image_paths = utils.generate_image_paths(test_desc, f'{train_path}test_images')

# Display the first three DICOM images
utils.display_dicom_images(train_image_paths)

# Display DICOM images with coordinates
study_id = "100206310"
study_folder = f'{train_path}/train_images/{study_id}'

image_paths = []
for series_folder in os.listdir(study_folder):
    series_folder_path = os.path.join(study_folder, series_folder)
    dicom_files = utils.load_dicom_files(series_folder_path)
    if dicom_files:
        image_paths.append(dicom_files[0])  # Add the first image from each series


utils.display_dicom_with_coordinates(image_paths, label)