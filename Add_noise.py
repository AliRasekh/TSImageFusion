from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
from trainers.fusion_trainer import FusionTrainer
from trainers.mmtm_trainer import MMTMTrainer
from trainers.daft_trainer import DAFTTrainer

from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from pathlib import Path
import torch

import os
import pandas as pd
import numpy as np
import argparse
import shutil
import random
from arguments import args_parser
import cv2

print('-----------------------------------------')
print(torch.cuda.get_device_name(0))
print('------------------------------------------')

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
print(args)

tasks = ['in-hospital-mortality', 'phenotyping']

noise_proportion = .5 
std_dev = 1  # Standard deviation of the Gaussian noise
print("############")
print(noise_proportion)
print(std_dev)

def copy_folders(task, file_path):
    
    for name in ['train', 'test']:
        print(name)
        source_folder = f'{args.ehr_data_dir}/{task}/{name}'
        destination_folder = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}/{name}'

    
        # Check if the source folder exists
        if os.path.exists(source_folder):
            # Create the destination folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            # Get a list of all files in the source folder
            files = os.listdir(source_folder)
        
            # Copy each file from the source folder to the destination folder
            for file in files:
                source_file_path = os.path.join(source_folder, file)
                destination_file_path = os.path.join(destination_folder, file)
                shutil.copy2(source_file_path, destination_file_path)  # Use copy2 to preserve metadata (timestamps, permissions, etc.)
        
            print("Folder copied successfully.")
        else:
            print("Source folder not found.")


        
        # Function to add Gaussian noise to numeric values
def add_noise(value):
    if pd.notnull(value) and np.issubdtype(type(value), np.number):
        # Generate Gaussian noise with mean 0 and standard deviation 1
        noise = np.random.normal(0, std_dev)
        # Add noise to the original value
        return value + noise
    else:
        return value
 
            
def ehr_data_noise():            

    for task in tasks:
        print(task)
        # Read your dataset into a DataFrame
        output_folder = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}'
        # Source folder path (the folder you want to copy)
        source_folder_train = f'{args.ehr_data_dir}/{task}/train'
        source_folder_test = f'{args.ehr_data_dir}/{task}/test'
        copy_folders(task,  f'{args.ehr_data_dir}/{task}')
    
        # Destination folder path (where you want to copy the folder)
        destination_folder_train = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}/train'
        destination_folder_test = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}/test'
    
        output_folder = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}'
        os.makedirs(output_folder, exist_ok=True)
        
        splits_labels_train = pd.read_csv(f'{args.ehr_data_dir}/{task}/train_listfile.csv')
        splits_labels_val = pd.read_csv(f'{args.ehr_data_dir}/{task}/val_listfile.csv')
        splits_labels_test = pd.read_csv(f'{args.ehr_data_dir}/{task}/test_listfile.csv')
    
        
        destination_file_path = f'{output_folder}/train_listfile.csv'
        splits_labels_train.to_csv(destination_file_path, index=False)
    
            
        destination_file_path = f'{output_folder}/val_listfile.csv'
        splits_labels_val.to_csv(destination_file_path, index=False)
    
        destination_file_path = f'{output_folder}/test_listfile.csv'
        splits_labels_test.to_csv(destination_file_path, index=False)
        
        
        
        # Read the CSV file into a DataFrame
        file_path = f'{output_folder}/train_listfile.csv'
        data = pd.read_csv(file_path)
        
        # Calculate the number of rows to select (10% of total rows)
        num_rows_to_select = int(noise_proportion * len(data))
        
        # Choose a random 10% of the rows from the DataFrame
        random_rows = data.sample(n=num_rows_to_select, random_state=random.seed())
        
        
        
         # Loop through the randomly selected rows and print the 'stay' column values
        for index, row in random_rows.iterrows():
            noise_file = row['stay']
            print(f"Randomly selected 'stay' value: {noise_file}")
            # Read the CSV file
            file_path_to_add_noise = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}/train/{noise_file}'
            data = pd.read_csv(file_path_to_add_noise)
           
            # Skip adding noise to the first column (assuming the first column is numeric)
            first_column = data.columns[0]
            non_first_columns = data.columns[1:]
            
            # Apply the add_noise function only to non-first numeric columns
            modified_data = data.copy()
            modified_data[non_first_columns] = modified_data[non_first_columns].applymap(add_noise)
            
            # Overwrite the original CSV file with the modified data
            modified_data.to_csv(file_path_to_add_noise, index=False)
        

def copy_folders_noise_on_all(task, file_path, percent_to_apply):
    
    for name in ['train', 'test']:
        print(name)
        source_folder = f'{args.ehr_data_dir}/{task}/{name}'
        destination_folder = f'{args.ehr_data_dir}_{percent_to_apply}_percent/{task}/{name}'

    
        # Check if the source folder exists
        if os.path.exists(source_folder):
            # Create the destination folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            # Get a list of all files in the source folder
            files = os.listdir(source_folder)
        
            # Copy each file from the source folder to the destination folder
            for file in files:
                source_file_path = os.path.join(source_folder, file)
                destination_file_path = os.path.join(destination_folder, file)
                shutil.copy2(source_file_path, destination_file_path)  # Use copy2 to preserve metadata (timestamps, permissions, etc.)
        
            print("Folder copied successfully.")
        else:
            print("Source folder not found.")


              
        
def ehr_data_noise_on_all(percent_to_apply):            
    
    #percent_to_apply = .2
    source_folder = f'{args.ehr_data_dir}/root'
    destination_folder = f'{args.ehr_data_dir}_{percent_to_apply}_percent'
    copy_folder(source_folder, destination_folder)


    for task in tasks:
        print(task)
        # Read your dataset into a DataFrame
        output_folder = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}/{task}'
        # Source folder path (the folder you want to copy)
        source_folder_train = f'{args.ehr_data_dir}/{task}/train'
        source_folder_test = f'{args.ehr_data_dir}/{task}/test'
        copy_folders_noise_on_all(task,  f'{args.ehr_data_dir}/{task}', percent_to_apply)
    
        # Destination folder path (where you want to copy the folder)
        destination_folder_train = f'{args.ehr_data_dir}_{percent_to_apply}_percent/{task}/train'
        destination_folder_test = f'{args.ehr_data_dir}_{percent_to_apply}_percent/{task}/test'
    
        output_folder = f'{args.ehr_data_dir}_{percent_to_apply}_percent/{task}'
        os.makedirs(output_folder, exist_ok=True)
        
        splits_labels_train = pd.read_csv(f'{args.ehr_data_dir}/{task}/train_listfile.csv')
        splits_labels_val = pd.read_csv(f'{args.ehr_data_dir}/{task}/val_listfile.csv')
        splits_labels_test = pd.read_csv(f'{args.ehr_data_dir}/{task}/test_listfile.csv')
    
        
        destination_file_path = f'{output_folder}/train_listfile.csv'
        splits_labels_train.to_csv(destination_file_path, index=False)
    
            
        destination_file_path = f'{output_folder}/val_listfile.csv'
        splits_labels_val.to_csv(destination_file_path, index=False)
    
        destination_file_path = f'{output_folder}/test_listfile.csv'
        splits_labels_test.to_csv(destination_file_path, index=False)
        
        
        overall_column_means = []
        overall_column_std_devs = []
        overall_column_max = []
        overall_column_min = []
        # Read the CSV file into a DataFrame
        
        file_path = f'{output_folder}/train_listfile.csv'
        data = pd.read_csv(file_path)
        
        num_rows_to_select = 1000
        
        random_rows = data.sample(n=num_rows_to_select, random_state=random.seed())
                
        
 
        # Loop through the randomly selected rows
        for index, row in random_rows.iterrows():
            noise_file = row['stay']
            print(f"Randomly selected 'stay' value: {noise_file}")
            
            # Read the CSV file
            file_path_to_add_noise = f'{args.ehr_data_dir}/{task}/train/{noise_file}'
            data = pd.read_csv(file_path_to_add_noise)
            
            
            # Calculate mean, standard deviation, maximum, and minimum for each column
            column_means = data.mean()
            column_std_devs = data.std()
            overall_column_means.append(column_means)
            overall_column_std_devs.append(column_std_devs)
            overall_column_max.append(data.max())
            overall_column_min.append(data.min())
            
        # Calculate the overall mean, standard deviation, maximum, and minimum
        overall_mean = pd.DataFrame(overall_column_means).mean()
        overall_std_dev = pd.DataFrame(overall_column_std_devs).mean()
        overall_max = pd.DataFrame(overall_column_max).max()
        overall_min = pd.DataFrame(overall_column_min).min()
        
        
        
        data_dir = f'{args.ehr_data_dir}_{percent_to_apply}_percent/{task}/test/'
        

        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
        
                # Read the CSV file
                data = pd.read_csv(file_path)
                num_rows_to_modify = int(percent_to_apply * len(data))
                rows_to_modify = np.random.choice(data.index, num_rows_to_modify, replace=False)
    
            else:
                continue
            for column in data.columns[1:]:
                if not column in column_means:
                    continue

                noise = np.random.normal(column_means[column], column_std_devs[column], len(rows_to_modify))
                modified_values = data.loc[rows_to_modify, column] + noise
                
                # Check if values exceed overall max or fall below overall min, and replace them
                modified_values[modified_values > overall_column_max[-1][column]] = overall_column_max[-1][column]
                modified_values[modified_values < overall_column_min[-1][column]] = overall_column_min[-1][column]
            
                data.loc[rows_to_modify, column] = modified_values
            data.to_csv(file_path, index=False)
            print(f"applied noise on {file_path}")
        

        data_dir = f'{args.ehr_data_dir}_{percent_to_apply}_percent/{task}/train/'
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
        
                # Read the CSV file
                data = pd.read_csv(file_path)
                num_rows_to_modify = int(percent_to_apply * len(data))
                rows_to_modify = np.random.choice(data.index, num_rows_to_modify, replace=False)
    
            else:
                continue
            for column in data.columns[1:]:
                if not column in column_means:
                    continue
                noise = np.random.normal(column_means[column], column_std_devs[column], len(rows_to_modify))
                modified_values = data.loc[rows_to_modify, column] + noise
                
                # Check if values exceed overall max or fall below overall min, and replace them
                modified_values[modified_values > overall_column_max[-1][column]] = overall_column_max[-1][column]
                modified_values[modified_values < overall_column_min[-1][column]] = overall_column_min[-1][column]
            
                data.loc[rows_to_modify, column] = modified_values
            data.to_csv(file_path, index=False)
            print(f"applied noise on {file_path}")


    
            
                    
            
        
def new_copy_folder(source_folder, destination_folder):
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Function to filter folders to be copied
        def filter_folders(folder):
            return os.path.basename(folder) != "resized"
        
        # Copy the entire folder and its contents, excluding "resized" folder
        shutil.copytree(source_folder, os.path.join(destination_folder, os.path.basename(source_folder)), ignore=shutil.ignore_patterns("resized"))
        
        print(f"Folder '{source_folder}' successfully copied to '{destination_folder}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
def copy_folder(source_folder, destination_folder):
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Copy the entire folder and its contents
        shutil.copytree(source_folder, os.path.join(destination_folder, os.path.basename(source_folder)))
        print(f"Folder '{source_folder}' successfully copied to '{destination_folder}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    pass

def copy_folder_2(source_folder, destination_folder, noise_file):
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Get the list of files in the source folder
        files = os.listdir(source_folder)
        
        # Copy each file to the destination folder and rename using noise_file variable
        for file in files:
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(destination_folder, f"{noise_file}.jpg")
            shutil.copy2(source_file, destination_file)
        
        print(f"Files from '{source_folder}' successfully copied to '{destination_folder}' with noise file '{noise_file}.jpg'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    pass


def add_gaussian_noise(image, mean=0, std=std_dev):
    print(std_dev)
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_noise_and_save(image_path):
    # Read the original image
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Unable to read the image from {image_path}")
        return
    
    # Add random Gaussian noise to the image
    noisy_image = add_gaussian_noise(original_image)

    # Overwrite the original image with the noisy version
    cv2.imwrite(image_path, noisy_image)
    
    
        
def cxr_data_noise():
    source_folder = f'{args.cxr_data_dir}'
    destination_folder = f'{args.cxr_data_dir}_{noise_proportion}_{std_dev}'

    copy_folder(source_folder, destination_folder)
    #copy_folder_2(source_folder, destination_folder)
    
        
    # Read the CSV file into a DataFrame
    file_path = f'{args.cxr_data_dir}/mimic-cxr-ehr-split.csv'
    data = pd.read_csv(file_path)
        
    # Calculate the number of rows to select (10% of total rows)
    num_rows_to_select = int(noise_proportion * len(data))
        
    # Choose a random 10% of the rows from the DataFrame
    random_rows = data.sample(n=num_rows_to_select, random_state=random.seed())
        
        # Loop through the randomly selected rows and print the 'stay' column values
    for index, row in random_rows.iterrows():
        noise_file = row['dicom_id']
        print(f"Randomly selected 'stay' value: {noise_file}")
        # Read the CSV file
        file_path_to_add_noise = f'{args.cxr_data_dir}_{noise_proportion}_{std_dev}/resized/{noise_file}.jpg'
        apply_noise_and_save(file_path_to_add_noise)
        #print("DONE!")
 

    pass




import glob
    
        
def cxr_data_noise_on_all(percent_to_apply):

    source_folder = f'{args.cxr_data_dir}'
    destination_folder = f'{args.cxr_data_dir}_{percent_to_apply}_percent_new'

    new_copy_folder(source_folder, destination_folder)
        
    # Read the CSV file into a DataFrame
    file_path = f'{args.cxr_data_dir}/mimic-cxr-ehr-split.csv'
    data = pd.read_csv(file_path)
        
    num_rows_to_select = 1000
        
    random_rows = data.sample(n=num_rows_to_select, random_state=random.seed())
    
    cumulative_mean = np.zeros(3)  # Assuming RGB images, adjust if using grayscale
    cumulative_std = np.zeros(3)
    cumulative_max = 0
    cumulative_min = 0
        
    
    for index, row in random_rows.iterrows():
        noise_file = row['dicom_id']
        print(f"Randomly selected 'dicom_id' value: {noise_file}")
        
        image_path = f'{args.cxr_data_dir}/resized/{noise_file}.jpg'
        img = cv2.imread(image_path)
        if img is None:
            continue
        print("####")
        print(img.shape)

        # Calculate mean, std, max, and pixel mean
        img_mean = np.mean(img, axis=(0, 1))
        img_std = np.std(img, axis=(0, 1))
        img_max = np.max(img)
        img_min = np.min(img)

        # Accumulate statistics
        cumulative_mean += img_mean
        cumulative_std += img_std
        cumulative_max = max(cumulative_max, img_max)
        cumulative_min = min(cumulative_min, img_min)

    # Calculate average statistics
    num_images = 1000
    average_mean = cumulative_mean / num_images
    average_std = cumulative_std / num_images
    print(f"average_mean : {average_mean}")
    print(f"average_std : {average_std}")
    
    
    data_dir = args.cxr_data_dir
    paths = glob.glob(f'{data_dir}/resized/**/*.jpg', recursive = True)
    CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
    filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

    metadata = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-metadata.csv')
    labels = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-chexpert.csv')
    labels[CLASSES] = labels[CLASSES].fillna(0)
    labels = labels.replace(-1.0, 0.0)
        
    splits = pd.read_csv(f'{data_dir}/mimic-cxr-ehr-split.csv')

    metadata_with_labels = metadata.merge(labels[CLASSES+['study_id'] ], how='inner', on='study_id')


    filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[CLASSES].values))
    filenames_loaded = splits['dicom_id'].values
    filenames_loaded = [filename  for filename in filenames_loaded if filename in filesnames_to_labels]
    print("number of files:")
    print(len(filenames_loaded))


    
    
    
    os.makedirs(f'{args.cxr_data_dir}_{percent_to_apply}_percent_new/mimic-cxr-jpg-2.0.0.physionet.org/resized', exist_ok=True)
    
    for noise_file in filenames_loaded:
        #noise_file = row['dicom_id']
        image_path = f'{args.cxr_data_dir}/resized/{noise_file}.jpg'
        img = cv2.imread(image_path)
        if img is None:
            print(image_path)
            print("missed")
            continue
        
        noisy_img = add_gaussian_noise_on_percent(img, percent_to_apply, cumulative_mean, cumulative_std, cumulative_max, cumulative_min)

        # Overwrite the original image with the noisy version
        cv2.imwrite(f'{args.cxr_data_dir}_{percent_to_apply}_percent_new/mimic-cxr-jpg-2.0.0.physionet.org/resized/{noise_file}.jpg', noisy_img)
        print(f'{args.cxr_data_dir}_{percent_to_apply}_percent_new/mimic-cxr-jpg-2.0.0.physionet.org/resized/{noise_file}.jpg')

        


    pass

import numpy as np



def add_gaussian_noise_on_percent(image, percent_pixels, mean, std, max_v, min_v):
    noisy_image = image.copy()

    # Get the dimensions of the image
    height, width, channels = noisy_image.shape

    # Calculate the number of pixels to modify
    num_pixels_to_modify = int(percent_pixels * height * width)

    # Randomly select 'num_pixels_to_modify' pixel indices
    pixel_indices_to_modify = np.random.choice(height * width, num_pixels_to_modify, replace=False)

    # Calculate the row and column indices of the selected pixels
    row_indices, col_indices = np.unravel_index(pixel_indices_to_modify, (height, width))

    # Generate Gaussian noise for all selected pixels in one go
    noise = np.random.normal(mean, std, (num_pixels_to_modify, channels))

    # Add the generated noise to the corresponding pixels
    noisy_image[row_indices, col_indices, :] += noise.astype(np.uint8)

    # Clip pixel values to the specified range
    noisy_image = np.clip(noisy_image, min_v, max_v)

    return noisy_image



def CLAHE_equalization():
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_noise_and_save(image_path):
    # Read the original image
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Unable to read the image from {image_path}")
        return
    
    # Add random Gaussian noise to the image
    noisy_image = add_gaussian_noise(original_image)

    # Overwrite the original image with the noisy version
    cv2.imwrite(image_path, noisy_image)
    
def apply_CLAHE_and_save(image_path):
    
    # Load the medical image (replace 'input_image.jpg' with the path to your medical image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a CLAHE object (Arguments are optional and can be adjusted based on image characteristics)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the image
    clahe_image = clahe.apply(image)

    # Save the CLAHE enhanced image
    cv2.imwrite(image_path, clahe_image)

        
def cxr_data_CLAHE():
    source_folder = f'{args.cxr_data_dir}'
    destination_folder = f'{args.cxr_data_dir}_CLAHE'

    copy_folder(source_folder, destination_folder)
    
        
    # Read the CSV file into a DataFrame
    file_path = f'{args.cxr_data_dir}/mimic-cxr-ehr-split.csv'
    data = pd.read_csv(file_path)
        
        
        # Loop through the randomly selected rows and print the 'stay' column values
    for index, row in data.iterrows():
        noise_file = row['dicom_id']
        print(f"Randomly selected 'stay' value: {noise_file}")
        # Read the CSV file
        file_path_to_add_noise = f'{args.cxr_data_dir}_CLAHE/resized/{noise_file}.jpg'
        apply_CLAHE_and_save(file_path_to_add_noise)
 

    pass


#file_path = f'{args.cxr_data_dir}/mimic-cxr-ehr-split.csv'
#data = pd.read_csv(file_path)

# Select rows with unique dicom_id
#unique_rows = data.drop_duplicates(subset='dicom_id', keep='first')

# Print or further process the DataFrame with unique dicom_id values
#print(unique_rows)


#ehr_data_noise_on_all(.6)
#cxr_data_noise_on_all(.6)

#cxr_data_CLAHE()
#cxr_data_noise()
#source_folder = f'{args.ehr_data_dir}/root'
#destination_folder = f'{args.ehr_data_dir}_{noise_proportion}_{std_dev}'

#copy_folder(source_folder, destination_folder)
#ehr_data_noise()



