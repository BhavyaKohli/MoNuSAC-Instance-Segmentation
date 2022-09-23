# MoNuSAC-Instance-Segmentation

Steps to install the mask-rcnn library and run test.ipynb in [`install.md`](./install.md):

Directory structure:
```
.
├── code
│   ├── monusac.ipynb
│   ├── monusac_utils.py
│   ├── README.md
│   └── sample_original
│       ├── A9K0_1.svs
│       ...
│       └── A9K0_3.xml
├── install.md
├── LICENSE
├── README.md
├── requirements.txt
└── test
    ├── dataset
    │   ├── train
    │   │   ├── ali-yahya-qh4kiGSV8pQ-unsplash.jpg
    │   │   ...
    │   │   └── via_project.json
    │   └── val
    │       ├── ethan-hu-nxGtkEBBF9k-unsplash.jpg
    │       ...
    │       └── zuriela-benitez-k7tJhSd1ZMY-unsplash.jpg
    ├── README.md
    └── test.ipynb
```

This repository contains my code for using the Mask R-CNN model [(Waleed Abdulla et al.)](https://github.com/matterport/Mask_RCNN) for performing cell nuclei instance segmentation and classification.)

The model trained for 15 epochs performs reasonably well on lymphocyte, epithelial and neutrophils but has trouble detecting/recognizing macrophages.

An interesting discovery - during dataset handling, I found that slide masks of shape HxWxN (N number of cells) could be saved in kBs using the `feather` library where it used hunreds of MBs using npy or any other format.