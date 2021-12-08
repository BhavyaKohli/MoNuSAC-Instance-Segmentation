# Importing required libraries
import os, glob, shutil, pathlib
import numpy as np                      
import matplotlib.pyplot as plt
import cv2

import openslide
import tifffile as tif
import xml.etree.ElementTree as ET
from skimage import draw

import tqdm.notebook as tq          # For creating progress bars for "for" loops

# Data locations
SLIDES_DIR = './data/slides/'
MASKS_DIR = './data/masks/'
ANNOTS_DIR = './data/annots/'

label_mapping = {1: 'Epithelial', 2: 'Lymphocyte', 3: 'Macrophage', 4: 'Neutrophil'}
'''# Lists with paths
SLIDES = [SLIDES_DIR + filename for filename in os.listdir('./data/slides')]

MASKS_TYPE = dict()
MASKS_ALL = []
for i in np.arange(1,5): 
    MASKS_TYPE[i] =  [MASKS_DIR + str(i) + '/' + filename for filename in [x[2] for x in os.walk('./data/masks')][i]]
    MASKS_ALL = MASKS_ALL + MASKS_TYPE[i]

ANNOTS = [ANNOTS_DIR + filename for filename in os.listdir('./data/annots')]'''

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
        for patient_loc in tq.tqdm(patients, unit = 'Patient'):
            unique = 0
            patient_name = patient_loc[len(data_path):] #Patient name
            patient_label = patient_name[8:12]
            
            sub_images = glob.glob(patient_loc + '/*.svs')

            for sub_image in sub_images:
                img = from_svs(sub_image)
                postfix = '_' + sub_image[-5]

                if slides: cv2.imwrite(os.path.join(dst_slides, patient_label + postfix + '.png'), img)
                #shutil.copy(sub_image, os.path.join(dst_slides, patient_label + postfix + '.svs'))

                xml_file_name  = sub_image[:-4]
                xml_file_name = xml_file_name+'.xml'
                if annots: shutil.copy(xml_file_name, os.path.join(dst_annots, patient_label + postfix + '.xml'))
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

                                mask_path = os.path.join(cell_type_dir, label + '_' + patient_label + postfix  + '.png')
                                if masks: cv2.imwrite(mask_path, mask)

    if slides: 
        num_slides = sum(1 for x in pathlib.Path(dst_slides).glob('**/*') if x.is_file())
        print(f"{num_slides} slides were copied to {dst_slides}")
    if masks: 
        num_masks = sum(1 for x in pathlib.Path(dst_masks).glob('**/*') if x.is_file())
        print(f"{num_masks} masks were created at {dst_masks}")
    if annots: 
        num_annots = sum(1 for x in pathlib.Path(dst_annots).glob('**/*') if x.is_file())
        print(f"{num_annots} annotations were copied to {dst_annots}")

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

def clean_plot(img, ax, title = None, cmap = None):
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

def show_mask(path, ax):
    '''
    Displays the mask saved as a .png file after loading it using the 
    matplotlib library.\n

    Arguments:\n
    path - string, path to the .png file\n
    ax - matplotlib axis object, the image is plotted on this axis\n

    Note: \n
    The image is displayed using the "Greys" colormap in matplotlib
    '''
    img = plt.imread(path)
    clean_plot(img, ax, "Mask" + " " + path[-10:-4] + "\n" + label_mapping[int(path[13])], "Greys")

def show_slide(path, ax = None, display = True, return_img = False):
    '''
    Displays the slide saved as a .png file after loading it using the 
    open cv library.\n

    Arguments:\n
    path - string, path to the .png file\n
    ax - matplotlib axis object, the image is plotted on this axis\n
    '''
    img = cv2.imread(path)
    if display: clean_plot(img, ax, f'Slide {path[-10:-4]}\nImage shape: {img.shape}')

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

def combine_masks(mask):
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
    print(depth)
    m = np.dsplit(mask, depth)[0]

    for i in tq.tqdm(range(depth)[1:]):
        m = m + np.dsplit(mask, depth)[i]
    return m

def collect_masks_for_id(patient_id):
    '''
    Collects all types of masks for the specified patient id from their 
    respective directories and merges them into one 2D array.\n
    '''
    ls = find_in_list(patient_id, MASKS_ALL)
    net_mask = tif_load(ls[0])
    for path in ls[1:]:
        net_mask += tif_load(path)
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
#############################################################################################

# CNN Architecture requirements

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



