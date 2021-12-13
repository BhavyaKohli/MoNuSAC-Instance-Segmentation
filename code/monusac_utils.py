import tensorflow as tf

# Silence deprecated-function warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Importing required libraries
import os, glob, shutil, pathlib
import numpy as np                      
import matplotlib.pyplot as plt
import cv2

import openslide
import tifffile as tif
import xml.etree.ElementTree as ET
from skimage import draw

# For creating progress bars for "for" loops
import tqdm as tq          

# Data locations
SLIDES_DIR = './data/slides/'
MASKS_DIR = './data/masks/'
ANNOTS_DIR = './data/annots/'

label_mapping = {1: 'Epithelial', 2: 'Lymphocyte', 3: 'Macrophage', 4: 'Neutrophil'}

# Functions
def from_svs(path, display = False, ax = None):
    '''
    Returns an array of shape (X, Y, 3) where (X, Y) are the dimensions of the slide
    stored in .svs format.\n

    Arguments:\n
    path - string, path to .svs file\n
    display -  bool, if True, plots the image created on the axis specified in "ax"\n
    ax - matplotlib axis object, used for plotting the image created\n

    Returns:\n
    img - numpy array of shape (X, Y, 3) containing the image which was stored
          in the .svs file  
    '''

    img = openslide.OpenSlide(path)
    img = np.array(img.get_thumbnail(size = img.level_dimensions[0]), dtype = np.uint8)

    if display:
        ax.imshow(img)
        ax.set_title(f'Slide {path[-10:-4]}\nImage shape: {img.shape}')
        ax.set_xticks([])
        ax.set_yticks([])
    return img

def generate_and_save_masks(src = './MoNuSAC_images_and_annotations/', 
                            dst = './data/',
                            dst_slides = 'slides', 
                            dst_masks = 'masks', 
                            dst_annots = 'annots', 
                            RESET = False, 
                            slides = True, 
                            masks = True, 
                            annots = True):
    '''
    Generates masks for the annotations in the .xml files and saves
    them as .tif files\n
    Creates three directories: slides/, masks/ and annots/ (for default dst paths)
    for the respective data.\n

    Arguments:\n
    src - string, contains the source directory containing the unzipped contents
          of the MoNuSAC dataset downloaded from https://monusac-2020.grand-challenge.org/Data/\n
    dst - string, contains directory of the destination folder in which all data created
          will be stored. A directory named "data" will be created in the root folder
          by default.\n
    dst_slides - string, contains the folder name for the destination of the slides. 
                 The slides are simply renamed, collected and copied to the folder 
                 created in the "dst" directory.\n
    dst_annots - string, contains the folder name for the destination of the annotations. 
                 The annotations are simply renamed, collected and copied to the folder 
                 created in the "dst" directory.\n
    dst_masks - string, contains the folder name for the destination of the masks. 
                The annotation files (found as .xml files) are read and all coordinates 
                are extracted. Masks are then created iteratively (one cell at a time) 
                using these coordinates and the skimage module.\n
    RESET - bool, if set to True, existing folders with the directories set using the above
            arguments will be deleted and recreated.\n
    slides, masks, annots - bool, if set to True, the files of their respective types will be
                            saved in their respective folders. If set to False, they will be 
                            loaded but not saved. (useful in the case of re-writing just the 
                            slides or just the annotations)\n

    Returns: \n
    patient_ids - list containing all patient_ids \n

    #########################################################################################
    Example usage:\n
        dst = './data/',\n
        num_patients = None,\n
        dst_slides = 'slides',\n
        dst_annots = '',\n
        dst_masks = '',\n
        RESET = False,\n
        slides = True,\n
        masks = False, annots = False\n

        result: Since reset is false, ignoring the current situation of the main destination 
                directory ('./data'), a folder named 'slides' will be created at './data/slides' 
                and will contain the .svs files extracted from ALL patients (num_patients = None).
                if the destination ('./data/slides') already exists, it will print the error 
                message and not do anything.\n
    #########################################################################################
    '''
    global label_mapping

    label_mapping = {
        1: 'Epithelial',
        2: 'Lymphocyte',
        3: 'Macrophage',
        4: 'Neutrophil'
    }

    dst_slides = os.path.join(dst, dst_slides)
    dst_masks = os.path.join(dst, dst_masks)
    dst_annots = os.path.join(dst, dst_annots)
    
    try: os.mkdir(dst)
    except: print(f"Directory {dst} already exists")

    if RESET:
        if slides: shutil.rmtree(dst_slides)
        if masks: shutil.rmtree(dst_masks)
        if annots: shutil.rmtree(dst_annots)

    data_path = src
    patients = [x[0] for x in os.walk(data_path)]
    try: 
        bool = True
        if slides: os.mkdir(dst_slides)
        if masks: os.mkdir(dst_masks)
        if annots: os.mkdir(dst_annots)
    except: 
        print("Destination directories already exist. Please rename or delete in case of conflicting names.")
        bool = False

    if bool:
        patient_ids = []      
        for patient_loc in tq.tqdm(patients, unit = 'Patient', ncols = 100):
            patient_name = patient_loc[len(data_path):] #Patient name
            patient_label = patient_name[8:12]
            

            sub_images = glob.glob(patient_loc + '/*.svs')

            for sub_image in sub_images:
                img = from_svs(sub_image)
                suffix = '_' + sub_image[-5]
                patient_ids.append(patient_label + suffix)

                if slides and bool: cv2.imwrite(os.path.join(dst_slides, patient_label + suffix + '.png'), img)
                #shutil.copy(sub_image, os.path.join(dst_slides, patient_label + suffix + '.svs'))

                '''if slides: 
                    img_path = os.path.join(dst_slides, patient_label + suffix + '.feather')
                    img_1d = list(img.flatten()) + list(img.shape)
                    feather.write_dataframe(pd.DataFrame(img_1d), img_path)'''

                xml_file_name  = sub_image[:-4]
                xml_file_name = xml_file_name+'.xml'
                if annots and bool: 
                    annots_path = os.path.join(dst_annots, patient_label + suffix + '.xml')
                    shutil.copy(xml_file_name, annots_path)
                
                tree = ET.parse(xml_file_name)
                root = tree.getroot()

                for k in range(len(root)):
                    
                    label_to_num = {
                        'Epithelial': '1',
                        'Lymphocyte': '2',
                        'Macrophage': '3',
                        'Neutrophil': '4',
                        'Description': 'X'
                    }

                    for child in root[k]:
                        unique = 0
                        for x in child:
                            r = x.tag
                            if r == 'Attribute':
                                label = x.attrib['Name']
                                mask = np.zeros((img.shape[0], img.shape[1], 1)) 

                                cell_type_dir = os.path.join(dst_masks, label_to_num[label])
                                try: os.mkdir(cell_type_dir)
                                except: pass

                            if r == 'Region':
                                vertices = x[1]
                                coords = np.zeros((len(vertices), 2))
                                for i, vertex in enumerate(vertices):
                                    coords[i][0] = vertex.attrib['X']
                                    coords[i][1] = vertex.attrib['Y']

                                vertex_row_coords = coords[:,0]
                                vertex_col_coords = coords[:,1]
                                fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, mask.shape)
                                unique += 0.4
                                mask[fill_row_coords, fill_col_coords] = unique

                        if masks and bool and np.sum(mask.flatten()) > 0: 
                            mask_path = os.path.join(cell_type_dir, label[:3].lower() + '_' + patient_label + suffix  + '.mask')
                            mask_1d = mask.flatten()
                            mask_1d_shape = mask.shape
                            mask_1d = list(mask_1d) + list(mask_1d_shape)
                            feather.write_dataframe(pd.DataFrame(mask_1d), mask_path)

    num_slides = sum(1 for x in pathlib.Path(dst_slides).glob('**/*') if x.is_file())
    print(f"{num_slides} slides are present in {dst_slides}")
    
    num_masks = sum(1 for x in pathlib.Path(dst_masks).glob('**/*') if x.is_file())
    print(f"{num_masks} masks are present in {dst_masks}")
    
    num_annots = sum(1 for x in pathlib.Path(dst_annots).glob('**/*') if x.is_file())
    print(f"{num_annots} annotations are present in {dst_annots}")

def tif_load(path, reshape = True):
    '''
    Returns a 1-D image as a numpy array after loading it from a .tif file
    using the tifffile library\n

    Arguments:\n
    path - string, path to the .tif file\n
    reshape - bool, if True, returns a reshaped array of shape (X, Y, 1) 
              where (X, Y) is the shape of the image stored in the .tif
              file\n
    
    Returns:\n
    img - numpy array, contains the image stored in the .tif file
    '''
    img = tif.imread(path)
    if reshape: img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

def clean_plot(img, ax, title = None, cmap = None, return_mask = False):
    '''
    Creates a plot showing an image without axis ticks.\n

    Arguments:\n
    img - array, contains an image as an array\n
    ax - matplotlib axis object, image will be plotted on this axis\n
    title - string, title of the plot\n
    cmap - string, should be a valid cmap option in the matplotlib
           library\n
    '''
    try:
        ax.imshow(img, cmap = cmap)
    except:
        ax.plot(img)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    #return ax

def show_mask(path, ax = None, cmap = "Greys", return_mask = True):
    '''
    Displays the mask saved as a .feather file after loading it using the 
    feather library.\n

    Arguments:\n
    path - string, path to the .feather file\n
    ax - matplotlib axis object, the image is plotted on this axis\n

    Note: \n
    The image is displayed using the "Greys" colormap in matplotlib
    by default
    '''
    df = feather.read_dataframe(path)
    mask_shape = [int(i) for i in df[-3:].values]
    mask = df[:-3].values.reshape(mask_shape)
    
    if ax != None:
        try: clean_plot(mask, ax, "Mask" + " " + path[-11:-5] + "\n" + label_mapping[int(path[11])], cmap)
        except: clean_plot(combine_masks(mask), ax, "Mask" + " " + path[-14:-5], cmap)

    if return_mask: return mask

def show_slide(path, read_mode = 'cv2', ax = None, return_img = True):
    '''
    Displays the slide saved as a .png file after loading it using the 
    open cv library.\n

    Arguments:\n
    path - string, path to the .png file\n
    ax - matplotlib axis object, the image is plotted on this axis\n
    '''

    if read_mode == 'plt': 
        img = plt.imread(path)
        title = path[-13:-4]
    elif read_mode == 'cv2': 
        img = cv2.imread(path)
        title = path[-10:-4]

    if ax != None: clean_plot(img, ax, f'Slide {title}\nImage shape: {img.shape}')

    if return_img: return img

def find_in_list(id, list):     
    '''
    Generates a list of all elements in the list "list" which contain "id"\n

    Arguments:\n
    id - string, contains the string the function will find in all elements
         of "list"\n
    list - list\n

    Returns:\n
    List containing all elements in the list "list" which contain "id"
    '''
    indices = []
    for i in range(len(list)):
        if id in list[i]: indices.append(i)

    return [list[i] for i in indices]

def num_cells(patient_id, annots_dir):
    '''
    Returns the total number of cells annotated for the slide with the given
    patient id\n

    Arguments:\n
    patient_id - string, uniquely determines the slide for which the number
                 of cells is desired.\n
    annots_dir - string, path to the annotations directory created using
                 the generate_and_save_masks function.   \n

    Returns:\n
    cell_count - integer, total number of cells in the image
    '''
    label_map = {
        'Epithelial': 1,
        'Lymphocyte': 2,
        'Macrophage': 3,
        'Neutrophil': 4
    }
    xml_file_path = annots_dir + patient_id + '.xml'
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    cell_count = [0,0,0,0,0]
    
    for k in range(len(root)):
        label = [x.attrib['Name'] for x in root[k][0]]
        label = label[0]
        
        for child in root[k]:
            for x in child:
                r = x.tag
                if r == 'Attribute':
                    label = x.attrib['Name']   
                
                if r == 'Region':
                    cell_count[label_map[label]] = cell_count[label_map[label]]+1

    return np.sum(cell_count)

def combine_masks(mask, progress = False):
    '''
    Takes a mask of shape (X,Y,N) where (X,Y) are the dimensions and
    N is the number of sub-masks and merges them into a single array 
    of shape (X,Y,1) for the purpose of visualization\n

    Arguments:\n
    mask - array, mask generated using the dataset.load_mask function\n

    Returns:\n
    m - array, combined mask generated from all sub-masks in the argument
        "mask"
    '''
    depth = mask.shape[2]
    if progress: print(depth)
    m = np.dsplit(mask, depth)[0]

    if progress:
        for i in tq.tqdm(range(depth)[1:], ncols = 100):
            m = m + np.dsplit(mask, depth)[i]
    else:
        for i in range(depth)[1:]: 
            m = m + np.dsplit(mask, depth)[i]
    return m

def collect_masks_for_id(patient_id, masks_list):
    '''
    Collects all types of masks for the specified patient id from their 
    respective directories and merges them into one 2D array.\n
    '''
    ls = find_in_list(patient_id, masks_list)
    net_mask = np.stack([show_mask(path) for path in ls], 2)[:,:,:,0]
    return net_mask

def split_mask_by_color(mask):
    '''
    Use only for smaller shapes ( < 512x512 )\n
    '''
    uniques = np.unique(mask.reshape(-1,1))
    masks = np.zeros(mask.shape)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == uniques[1]: masks[i][j] = 1

    for value in uniques[2:]:
        temp = np.zeros(mask.shape)
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j] == value: temp[i][j] = 1
        masks = np.concatenate((masks, temp), axis = 2) 
    
    return masks

def modify_mask_values(mask, value):
    '''
    Returns a mask in which all non-zero values of the input mask are replaced 
    by the argument "value"\n
    '''
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] > 0: mask[i][j] = value
    return mask

#############################################################################################
# Cropping slides and masks based on number of cells
#############################################################################################

import pandas as pd
import feather
from PIL import Image
from collections import Counter

def get_mask(patient_id, slides_dir = './data/slides/', annots_dir = './data/annots/'):
    global label_map
    
    label_map = {
        'Epithelial': 1,
        'Lymphocyte': 2,
        'Macrophage': 3,
        'Neutrophil': 4,
        'Description': 'X'
    }

    annotations_path = os.path.join(annots_dir, f'{patient_id}.xml')

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    img_path = os.path.join(slides_dir, f'{patient_id}.png')
    img = show_slide(img_path, return_img = True, ax = None)
    masks = dict()
    labels = []
    cell_count = [0,0,0,0,0]

    count = 0
    for k in range(len(root)):
        for child in root[k]:
            for x in child:
                r = x.tag
                if r == 'Attribute':
                    label = x.attrib['Name']
                    label_id = label_map[label]

                if r == 'Region':
                    binary_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype = np.uint8)

                    vertices = x[1]
                    coords = np.zeros((len(vertices), 2))
                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib['X']
                        coords[i][1] = vertex.attrib['Y']

                    vertex_row_coords = coords[:,0]
                    vertex_col_coords = coords[:,1]

                    fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, binary_mask.shape)

                    binary_mask[fill_row_coords, fill_col_coords] = 1
                    masks[count] = (binary_mask, label_id)
                    labels.append(label_id)
                    cell_count[label_id] += 1
                    count += 1
                    
    keys = list(masks.keys())
    net_mask = np.stack([masks[key][0] for key in keys], axis = 2)
    net_mask = net_mask.reshape(net_mask.shape[:3])
    net_labels = [masks[key][1] for key in keys]
    '''for key in keys[1:]:
        net_mask = np.concatenate((net_mask, masks[key][0]), axis = 2)
        net_labels.append(masks[key][1])'''

    #print(net_mask.shape)
    #print(np.dsplit(net_mask, net_mask.shape[2]).shape)
    
    return net_mask.astype(np.bool_), np.array(net_labels), cell_count
    
def crop_image(slide, height, width):
    img = Image.fromarray(slide)
    img_width, img_height = img.size
    for i in range(img_height//height):
        for j in range(img_width//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield img.crop(box)

def crop_mask(mask, height, width):
    masks = dict()
    net_masks = dict()
    
    for i in range(mask.shape[2]):
        masks[i] = dict()
        slice = mask[:,:,i]
        img = Image.fromarray(slice)
        img_width, img_height = img.size
        for j in range(img_height//height):
            for k in range(img_width//width):
                box = (k*width, j*height, (k+1)*width, (j+1)*height)
                masks[i][f'{j}{k}'] = np.array(img.crop(box))
    
    net_masks = dict()
    keys = list(masks.keys())
    subkeys = list(masks[keys[0]].keys())
    #[m[:,:,i] for i in range(m.shape[2])] 
    for i in range(len(subkeys)):    
        net_masks[subkeys[i]] = np.stack(pd.DataFrame(masks).iloc[i].values, 2) 
    
    return net_masks

def filter_cropped_mask(mask, mask_labels, type):
    df = pd.DataFrame([Counter(mask[:,:,depth].flatten()) for depth in range(mask.shape[2])])
    
    if type == 'all': 
        ls = [not (True in mask[:,:,i][0,:] or True in mask[:,:,i][-1,:] or True in mask[:,:,i][:,0] or True in mask[:,:,i][:,-1] or True not in mask[:,:,i].flatten()) for i in range(mask.shape[2])]   
    elif type == 'edges': 
        ls = [not (True in mask[:,:,i][0,:] or True in mask[:,:,i][-1,:] or True in mask[:,:,i][:,0] or True in mask[:,:,i][:,-1]) for i in range(mask.shape[2])]
    elif type == 'zeros': 
        ls = [True in mask[:,:,i].flatten() for i in range(mask.shape[2])]
    elif type == None:
        return mask, mask_labels
    else:
        print(f"Unknown argument for \"type\" {type}")
    
    if True in ls:
        df = df[ls]
        return_mask = np.stack([mask[:,:,i] for i in df.index], axis = 2)
        return_labels = [mask_labels[i] for i in df.index]

        return return_mask, return_labels
    else:
        return None, None

def split(image, mask, mask_labels, mask_filter_type, patient_id, train, out_dir, height, width):
    """
    Splits the input image and mask into smaller parts of shape determined
    by the "height" and "width" arguments \n

    Arguments: \n
    image - numpy array of shape (X,Y,3), contains the image of a slide 
            as an array \n
    mask - numpy array of shape (X,Y,N), contains the corresponding mask 
           of the image argument \n
    mask_labels - list/1d array, contains labels corresponding to each layer
                  of masks stored in mask \n
    patient_id - string, the corresponding patient id for the image and mask
                 described above \n
    train - bool, determines whether to create the "train" or "val" sub-directory \n
    height, width - integers, determine the shape of the output images \n

    All smaller images are saved using the _00, _01,... suffixes after the 
    patient_id.
    """
    image_dir = os.path.join(out_dir, 'train/slides') if train else os.path.join(out_dir, 'val/slides')
    mask_dir = os.path.join(out_dir, 'train/masks') if train else os.path.join(out_dir, 'val/masks')

    cropped_masks = crop_mask(mask, height, width)
    exceptions = []

    for i, key in zip(range(len(cropped_masks)), cropped_masks):
        mask_path = os.path.join(mask_dir, patient_id + '_' + str(i).zfill(2) + '.mask')
        mask_label_path = os.path.join(mask_dir, patient_id + '_' + str(i).zfill(2) + '.label')
        mask_shape_path = os.path.join(mask_dir, patient_id + '_' + str(i).zfill(2) + '.shape')
        mask = cropped_masks[key]
        
        filtered_mask, filtered_labels = filter_cropped_mask(mask, mask_labels, mask_filter_type)
        
        if type(filtered_mask) != type(None):
            mask_1d = np.append(filtered_mask.flatten(), filtered_mask.shape)
            feather.write_dataframe(pd.DataFrame(mask_1d), mask_path)
            feather.write_dataframe(pd.DataFrame(filtered_labels), mask_label_path)
        else:
            print(f"{patient_id}_{str(i).zfill(2)} has no mask")
            exceptions.append(patient_id)

    if patient_id not in exceptions:
        for i, piece in enumerate(crop_image(image, height, width)):
            img = Image.new('RGB', (width, height))
            img.paste(piece)
            img_path = os.path.join(image_dir , patient_id + '_' + str(i).zfill(2) + '.png')
            cv2.imwrite(img_path, np.array(img))

    '''        df = pd.DataFrame([Counter(mask[:,:,depth].flatten()) for depth in range(mask.shape[2])])
            ls = [not (True in mask[:,:,i][0,:] or True in mask[:,:,i][-1,:] or True in mask[:,:,i][:,0] or True in mask[:,:,i][:,-1] or True not in mask[:,:,i].flatten()) for i in range(mask.shape[2])]

            if True in ls: 
                df = df[ls]
                filtered_mask = np.stack([mask[:,:,i] for i in df.index], axis = 2)
                filtered_labels = [mask_labels[i] for i in df.index]

                mask_1d = np.append(filtered_mask.flatten(), filtered_mask.shape)
                feather.write_dataframe(pd.DataFrame(mask_1d), mask_path)
                feather.write_dataframe(pd.DataFrame(filtered_labels), mask_label_path)

            else:
                print(f"{patient_id + '_' + str(i).zfill(2)} has no mask")
    '''

def create_train_test_data(slides_dir, train, train_size, out_dir, patient_ids, RESET = False):
    """
    Creates the following directories in the root directory
    1. A directory defined by the "out_dir"  \n
    2. Sub-directories named "train" and "val", dependent on the "train"
       argument inside the created out_dir directory \n
    3. Sub-sub-directories "slides" and "masks" for storing their respective
       types of data \n

    Slides are read from the "slides_dir" directory which were created using
    the generate_and_save_masks function \n

    Arguments: \n
    slides_dir - string, location of all slides in .png format, created using
                 as described above \n
    train - bool, when True, creates the "train" sub-directory and when False,
            creates the "val" sub-directory \n 
    train_size - float value between 0 and 1, determines the relative size of 
                 the training set compared to the complete dataset \n 
    out_dir - string, location where the outputs will be saved \n
    RESET - bool, when True, deletes the existing "out_dir" directory along
            with its contents and creates it from scratch \n
    """  
    bool = True   

    try: os.mkdir(out_dir)
    except: print(f"Directory {out_dir} exists")

    dir = 'train' if train else 'val'
    dir = os.path.join(out_dir, dir)

    if RESET: shutil.rmtree(dir)

    try: os.mkdir(dir)
    except: print(f"Directory {dir} exists")

    image_dir = os.path.join(dir, 'slides')  
    mask_dir = os.path.join(dir, 'masks')      

    try:
        os.mkdir(image_dir)
        os.mkdir(mask_dir)
    except:
        bool = False

    train_len = int(train_size * len(patient_ids))
    ids = patient_ids[:train_len] if train else patient_ids[train_len + 1:]

    if bool:
        for patient_id in tq.tqdm(ids, ncols = 100):
            slide_path = os.path.join(slides_dir, f'{patient_id}.png')
            slide = show_slide(slide_path, read_mode = 'cv2', return_img = True, ax = None)
            mask, mask_labels, n = get_mask(patient_id)
            n = np.sum(n)

            x, y = slide.shape[:2]

            if n >= 300:
                split(slide, mask, mask_labels, 'zeros', patient_id, train, out_dir, x//4, y//4)
            elif n < 300 and n >= 150:
                split(slide, mask, mask_labels, 'zeros', patient_id, train, out_dir, x//2, y//2)
            else:
                split(slide, mask, mask_labels, 'zeros', patient_id, train, out_dir, x, y)

    num_slides = sum(1 for x in pathlib.Path(dir).glob('**/*.png'))
    print(f"{num_slides} slides were created at {dir}")

    num_masks = sum(1 for x in pathlib.Path(dir).glob('**/*.mask'))
    print(f"{num_masks} masks were created at {dir}")

#############################################################################################
# CNN Architecture requirements
#############################################################################################

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential

def create_model(weights_path):
    '''
    Returns a CNN model with pre-loaded weights.\n
    The model was trained using 512x512 sized images and so any inputs to it should
    be resized appropriately using cv2.resize(image, (512,512))
    '''
    model = Sequential([
        Conv2D(10, (5,5), input_shape = (512, 512, 1), data_format = "channels_last"),
        MaxPool2D((3,3)),
        Conv2D(5, (3,3)),
        MaxPool2D((3,3)),
        Flatten(),
        Dropout(0.2),
        Dense(8),
        Dense(1)
    ])
    return model.load_weights(weights_path)

def estimate_num_cells(image, weights_path):
    '''
    Estimates the number of cells in the argument image using a CNN
    model created using the create_model function with weights 
    specified using the weights_path argument.
    '''
    model = create_model(weights_path)
    if image.shape[:2] != (512,512):
        image = cv2.resize(image, (512,512))
    return model.predict(image)[0][0]

