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

# Lists with paths
SLIDES = [SLIDES_DIR + filename for filename in os.listdir('./data/slides')]

MASKS_TYPE = dict()
#MASKS_ALL = []
#for i in np.arange(1,5): 
#    MASKS_TYPE[i] =  [MASKS_DIR + str(i) + '/' + filename for filename in [x[2] for x in os.walk('./data/masks')][i]]
#    MASKS_ALL = MASKS_ALL + MASKS_TYPE[i]

ANNOTS = [ANNOTS_DIR + filename for filename in os.listdir('./data/annots')]

# Functions
def from_svs(path, display = False, ax = None):
    '''
    Returns an array of shape (X, Y, 3) where (X, Y) are the dimensions of the slide
    stored in .svs format.

    Arguments:
    path - string, path to .svs file
    display -  bool, if True, plots the image created on the axis specified in "ax"
    ax - matplotlib axis object, used for plotting the image created

    Returns:
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
    them as .tif files
    Creates three directories: slides/, masks/ and annots/ (for default dst paths)
    for the respective data.

    Arguments:
    src - string, contains the source directory containing the unzipped contents
          of the MoNuSAC dataset downloaded from https://monusac-2020.grand-challenge.org/Data/
    dst - string, contains directory of the destination folder in which all data created
          will be stored. A directory named "data" will be created in the root folder
          by default.
    dst_slides - string, contains the folder name for the destination of the slides. 
                 The slides are simply renamed, collected and copied to the folder 
                 created in the "dst" directory.
    dst_annots - string, contains the folder name for the destination of the annotations. 
                 The annotations are simply renamed, collected and copied to the folder 
                 created in the "dst" directory.
    dst_masks - string, contains the folder name for the destination of the masks. 
                The annotation files (found as .xml files) are read and all coordinates 
                are extracted. Masks are then created iteratively (one cell at a time) 
                using these coordinates and the skimage module.
    RESET - bool, if set to True, existing folders with the directories set using the above
            arguments will be deleted and recreated.
    slides, masks, annots - bool, if set to True, the files of their respective types will be
                            saved in their respective folders. If set to False, they will be 
                            loaded but not saved. (useful in the case of re-writing just the 
                            slides or just the annotations)
    #########################################################################################
    Example usage:
        dst = './data/',
        num_patients = None,
        dst_slides = 'slides',
        dst_annots = '',
        dst_masks = '',
        RESET = False,
        slides = True,
        masks = False, annots = False

        result: Since reset is false, ignoring the current situation of the main destination 
                directory ('./data'), a folder named 'slides' will be created at './data/slides' 
                and will contain the .svs files extracted from ALL patients (num_patients = None).
                if the destination ('./data/slides') already exists, it will print the error 
                message and not do anything.
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

                if slides: shutil.copy(sub_image, os.path.join(dst_slides, patient_label + postfix + '.svs'))

                xml_file_name  = sub_image[:-4]
                xml_file_name = xml_file_name+'.xml'
                if annots: shutil.copy(xml_file_name, os.path.join(dst_annots, patient_label + postfix + '.xml'))
                tree = ET.parse(xml_file_name)
                root = tree.getroot()

                for k in range(len(root)):

                    label_postfix = {
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
                                binary_mask = np.zeros((img.shape[0], img.shape[1], 1)) 

                                cell_type_dir = os.path.join(dst_masks, label_postfix[label])
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
                                fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, binary_mask.shape)
                                unique += 0.4
                                binary_mask[fill_row_coords, fill_col_coords] = unique
                                mask_path = os.path.join(cell_type_dir, label + '_' + patient_label + postfix  + '.tif')
                                if masks: cv2.imwrite(mask_path, binary_mask)

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
    using the tifffile library

    Arguments:
    path - string, path to the .tif file
    reshape - bool, if True, returns a reshaped array of shape (X, Y, 1) 
              where (X, Y) is the shape of the image stored in the .tif
              file
    
    Returns:
    img - numpy array, contains the image stored in the .tif file
    '''
    img = tif.imread(path)
    if reshape: img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

def tif_show(path, ax = plt.subplots()[1]):
    '''
    Displays the image stored in a .tif file after loading it using the 
    tifffile library.

    Arguments:
    path - string, path to the .tif file
    ax - matplotlib axis object, the image is plotted on this axis

    Note: 
    The image is displayed using the "Greys" colormap in matplotlib
    '''
    img = tif.imread(path)
    ax.imshow(img, cmap = "Greys")
    ax.set_title("Mask" + " " + path[-10:-4] + "\n" + label_mapping[int(path[13])])
    ax.set_xticks([])
    ax.set_yticks([])

def find_in_list(id, list):
    '''
    Generates a list of all elements in the list "list" which contain "id"

    Arguments:
    id - string, contains the string the function will find in all elements
         of "list"
    list - list

    Returns:
    List containing all elements in the list "list" which contain "id"
    '''
    indices = []
    for i in range(len(list)):
        if id in list[i]: indices.append(i)

    return [list[i] for i in indices]

def num_cells(patient_id, annots_dir = ANNOTS_DIR):
    '''
    Returns the total number of cells annotated for the slide with the given
    patient id

    Arguments:
    patient_id - string, uniquely determines the slide for which the number
                 of cells is desired.
    annots_dir - string, path to the annotations directory created using
                 the generate_and_save_masks function.   

    Returns:
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
    
    #Generate binary mask for each cell-type                         
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
    depth = mask.shape[2]
    print(depth)
    m = np.dsplit(mask, depth)[0]

    for i in tq.tqdm(range(depth)[1:]):
        m = m + np.dsplit(mask, depth)[i]
    return m








