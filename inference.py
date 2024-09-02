import seaborn as sns

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import glob
import json
import collections
import torch
import torch.nn as nn

import pydicom as dicom
import matplotlib.patches as patches

from matplotlib import animation, rc
import pandas as pd

import pydicom as dicom # dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# read data
train_path = '/mnt/rsna/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')

# Function to generate image paths based on directory structure
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        images = os.listdir(series_dir)
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

# Generate image paths for train and test data
train_image_paths = generate_image_paths(train_desc, f'{train_path}/train_images')
test_image_paths = generate_image_paths(test_desc, f'{train_path}/test_images')

# Define function to reshape a single row of the DataFrame
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    
    for column, value in row.items():
        if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)
    
    return pd.DataFrame(data)

# Reshape the DataFrame for all rows
new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)

# Merge the dataframes on the common columns
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on='series_id', how='inner')

# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id','study_id'], how='inner')
# Display the first few rows of the final merged dataframe
final_merged_df.head(5)


# Create the row_id column
final_merged_df['row_id'] = (
    final_merged_df['study_id'].astype(str) + '_' +
    final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
    final_merged_df['level'].str.lower().str.replace('/', '_')
)

# Create the image_path column
final_merged_df['image_path'] = (
    f'{train_path}train_images/' + 
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.dcm'
)

# Note: Check image path, since there's 1 instance id, for 1 image, but there's many more images other than the ones labelled in the instance ID. 

# Display the updated dataframe
print(final_merged_df.head(5))

# Define the base path for test images
base_path = '/mnt/rsna/test_images/'

# Function to get image paths for a series
def get_image_paths(row):
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [os.path.join(series_path, f) for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
    return []

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []

# Expand the dataframe by adding new rows for each file path
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new dataframe from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)

# Display the resulting dataframe
print(expanded_test_desc.head(5))

# change severity column labels
#Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'}
final_merged_df['severity'] = final_merged_df['severity'].map({'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

test_data = expanded_test_desc
train_data = final_merged_df

import os

# Define a function to check if a path exists
def check_exists(path):
    return os.path.exists(path)

# Define a function to check if a study ID directory exists
def check_study_id(row):
    study_id = row['study_id']
    path = f'{train_path}/train_images/{study_id}'
    return check_exists(path)

# Define a function to check if a series ID directory exists
def check_series_id(row):
    study_id = row['study_id']
    series_id = row['series_id']
    path = f'{train_path}/train_images/{study_id}/{series_id}'
    return check_exists(path)

# Define a function to check if an image file exists
def check_image_exists(row):
    image_path = row['image_path']
    return check_exists(image_path)

# Apply the functions to the train_data dataframe
train_data['study_id_exists'] = train_data.apply(check_study_id, axis=1)
train_data['series_id_exists'] = train_data.apply(check_series_id, axis=1)
train_data['image_exists'] = train_data.apply(check_image_exists, axis=1)

# Filter train_data
train_data = train_data[(train_data['study_id_exists']) & (train_data['series_id_exists']) & (train_data['image_exists'])]

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
tqdm.pandas()

import gc

#############################################################
#                                                           #
#   Segmentation using Unet trained from Zenodo Dataset     #
#                                                           #
#############################################################

BASE_PATH  = '/mnt/lumbar/dataset.csv'
CKPT_DIR = './'

class CFG:
    seed          = 101
    debug         = False # set debug=False for Full Training
    exp_name      = 'Baseline'
    comment       = 'unet-efficientnet_b1-224x224'
    model_name    = 'Unet'
    backbone      = 'efficientnet-b1'
    train_bs      = 64
    valid_bs      = train_bs*2
    img_size      = [224, 224]
    epochs        = 15
    lr            = 2e-3
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+50
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = max(1, 32//train_bs)
    n_fold        = 5
    num_classes   = 3
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr           = 0.45
    ttas          = [0]

def load_img(path):
    if(path[-4:]=='.dcm'):
        img = pydicom.dcmread(path).pixel_array
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk/=255.0
    return msk
    

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]

    plt.axis('off')

import cupy as cp

def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = cp.array(msk)
    pixels = msk.flatten()
    pad    = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs   = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        msk = cv2.resize(msks[idx], 
                         dsize=(width, height), 
                         interpolation=cv2.INTER_NEAREST) # back to original shape
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=False, transforms=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.ids        = df['study_id'].tolist()
        if 'msk_path' in df.columns:
            self.msk_paths  = df['mask_path'].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        id_       = self.ids[index]
        img = []
        img = load_img(img_path)
        h, w = img.shape[:2]
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), id_, h, w

data_transforms = {
    "train": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
# #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=1.0)
        ], p=0.25),
#         A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
#                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),
    
    "valid": A.Compose([
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
}

import segmentation_models_pytorch as smp

def build_model():
    model = smp.Unet(
        encoder_name=CFG.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=CFG.num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(CFG.device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks = []; imgs = [];
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx, (img, ids, heights, widths) in enumerate(tqdm(test_loader, total=len(test_loader), desc='Infer ')):
        img = img.to(CFG.device, dtype=torch.float) # .squeeze(0)
        size = img.size()
        msk = []
        msk = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32)
        for path in model_paths:
            model = load_model(path)
            out   = model(img) # .squeeze(0) # removing batch axis
            out   = nn.Sigmoid()(out) # removing channel axis
            msk+=out/len(model_paths)
        msk = (msk.permute((0,2,3,1))>thr).to(torch.uint8).cpu().detach().numpy() # shape: (n, h, w, c)
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
        if idx<num_log:
            img = img.permute((0,2,3,1)).cpu().detach().numpy()
            imgs.append(img[:10])
            msks.append(msk[:10])
        del img, msk, out, model, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks

def predict_mask(condition='Sagittal T2/STIR'):
    test_df = train_data.loc[train_data['series_description'] == condition]

    test_dataset = BuildDataset(test_df, transforms=data_transforms['valid'])
    test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, 
                              num_workers=4, shuffle=False, pin_memory=False)
    model_paths  = glob.glob(f'{CKPT_DIR}/best_epoch*.bin')
    pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)
    
    ########### Visualization ###################
    for img, msk in zip(imgs[0][:5], msks[0][:5]):
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 3, 1); plt.imshow(img, cmap='bone');
        plt.axis('OFF'); plt.title('image')
        plt.subplot(1, 3, 2); plt.imshow(msk*255); plt.axis('OFF'); plt.title('mask')
        plt.subplot(1, 3, 3); plt.imshow(img, cmap='bone'); plt.imshow(msk*255, alpha=0.4);
        plt.axis('OFF'); plt.title('overlay')
        plt.tight_layout()
        plt.savefig('./predict_zenodo.png', format='png', pad_inches=0.1)

predict_mask(condition = 'Sagittal T2/STIR')

#############################################################
#                                                           #
#   Segmentation using Unet trained from Spider Dataset     #
#                                                           #
#############################################################
import numpy as np 
import pandas as pd 
import os
from pathlib import Path
from PIL import Image
from matplotlib.patches import Rectangle

from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from segmentation_models_pytorch import Unet
gc.collect()

#transforms
newsize = (256, 256)
#dataset
fold = 1
#dataloader
batch_size = 64
num_workers = 4
#model
num_classes = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#run
epochs = 100
learning_rate = 1e-3

TRAIN = False #or False for inference only

model = Unet(
  encoder_name="resnet34",  # Choose encoder (e.g. resnet18, efficientnet-b0)
  classes=num_classes,  # Number of output classes
  in_channels=3  # Number of input channels (e.g. 3 for RGB)
)

class SEGDataset(Dataset):
    def __init__(self, df, mode, transforms=None):
        self.df = df.reset_index()
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row.image_path

        # Open image
        image = pydicom.dcmread(image_path).pixel_array
        image = Image.fromarray(image)
        if image.mode != 'RGB':  # Ensure image is RGB
            image = image.convert('RGB')
        image = np.asarray(image)
        if (image > 1).any():  # Normalize if pixel values are between 0-255
            image = image / 255.0
        mask = np.zeros((image.shape[0], image.shape[1]))
        
        if(self.mode=='train'):
            mask_path = os.path.join(mask_dir, row.image)
            # Open mask
            mask = Image.open(mask_path)
            mask = np.asarray(mask)
            assert mask.max() < num_classes, f"Mask value {mask.max()} exceeds number of classes {num_classes}"

            # Apply transformations
            if self.transforms is not None:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            # Create one layer for each label
            mask = torch.as_tensor(mask).long()
            mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(2,0,1).float()
            #mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(0,3,1,2).squeeze(0).float()
        else:
            if self.transforms is not None:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

    # Create one layer for each label
        mask = torch.as_tensor(mask).long()
        mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(2,0,1).float()
        #mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(0,3,1,2).squeeze(0).float()    
    # Convert image to tensor
        image = torch.as_tensor(image).float()

        return image, mask

transforms_train = A.Compose([
    A.Resize(newsize[0], newsize[1]),
    A.HorizontalFlip(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
])

transforms_valid = A.Compose([
    A.Resize(newsize[0], newsize[1]),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
])

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_iou=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # Cross-Entropy Loss
        ce_loss = self.cross_entropy_loss(inputs, targets)

        # IoU Loss
        # Apply softmax to the inputs to get probabilities
        probs = F.softmax(inputs, dim=1)

        intersection = torch.sum(probs * targets, dim=(2, 3))
        union = torch.sum(probs + targets, dim=(2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean()

        # Combine losses
        loss = self.weight_ce * ce_loss + self.weight_iou * iou_loss
        return loss

test = train_data.loc[train_data['series_description'] == 'Sagittal T2/STIR'].reset_index(drop=True)
test_df = test[['image_path']]

dataset_valid = SEGDataset(test_df, 'valid',  transforms_valid)

val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=20, shuffle=False, num_workers=num_workers, pin_memory=False)
criterion = CombinedLoss()
model.to(device)

if TRAIN:
    run(train_loader, val_loader, model, learning_rate, criterion, epochs, device)
else:
    model.load_state_dict(torch.load("./simple_unet.pth"))

import matplotlib.pyplot as plt

def inference(model, dataloader, device, num_samples=16):
    model.eval()
    images_batch = []
    preds_batch = []
    
    with torch.no_grad():
        print('inference start')
        for images, _ in dataloader:
            print(images.shape)
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            images_batch.append(images.cpu())
            preds_batch.append(preds.cpu())
            
            if len(images_batch) * images.size(0) >= num_samples:
                break
#     print(len(images_batch))
    images_batch = torch.cat(images_batch)[:num_samples]
    preds_batch = torch.cat(preds_batch)[:num_samples]
    
    return images_batch, preds_batch


# Define a color map with fixed colors for each label
def get_label_colors(num_classes):
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    return colors

label_dict = {1 : '1: L5', 2 : '2: L4', 3 : '3: L3', 4 : '4: L2', 5 : '5: L1', 6 : '6: T12',
                7 : '7: unknown', 8 : '8: unknown', 9 : '9: unknown',
                10: '10: spinal canal', 11: '11: L5-S1', 12: '12: L4-L5', 13: '13: L3-L4',
                14: '14: L2-L3', 15: '15: L1-L2', 16: '16: T12-L1',
                17: '17: unknown', 18: '18: unknown', 19: '19: unknown'
             }

def visualize_predictions(images, masks, num_classes=20, num_samples=16):
#     print(images[0])
    num_samples = min(num_samples, len(images))
    plt.figure(figsize=(25, 20))
    
    label_colors = get_label_colors(num_classes)
    
    for i in range(num_samples):
        plt.subplot(4, 8, i * 2 + 1)
        im = images[i].numpy()
        im = np.transpose(im, (1, 2, 0))
        #denormalize
        im = ((im * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255
        plt.imshow(im)
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(4, 8, i * 2 + 2)
        mask = masks[i].numpy()

        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        for label in range(num_classes):
            color_mask[mask == label] = label_colors[label][:3] * 255
        
        plt.imshow(color_mask.astype(np.uint8))
        plt.title("Predicted Mask")
        plt.axis('off')
    ## for legend
    plt.subplot(4, 8, 4*8)
    handles = [Rectangle((0,0),1,1, color=label_colors[i][:3]) for i in range(1, num_classes)]
    labels = list(label_dict.values())
    plt.legend(handles,labels, loc="upper right", fontsize=20)
    plt.axis('off')

    plt.savefig('./predict_spider.png', format='png', pad_inches=0.1)
model.eval()
model.to(device)

images, masks = inference(model, val_loader, device, num_samples=15)
visualize_predictions(images, masks, num_samples=15)