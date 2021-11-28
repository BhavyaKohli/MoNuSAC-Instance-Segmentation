# MoNuSAC-Instance-Segmentation

Steps to install the mask-rcnn library and run test.ipynb:

IMPORTANT: Tested with python version 3.7.x. If your base version is not 3.7, refer to [pyenv-win](https://github.com/pyenv-win/pyenv-win) for installing `pyenv`, a convenient way of managing multiple python versions on windows.

Basic setup for a virtual environment:

1. Navigate to the root directory ((path)/monusac-instance-segmentation) in cmd
2. Run `python -m venv venv_name`
3. For using jupyter, run `ipython kernel install --user --name=venv_name` <br> (For removing the kernel, run `jupyter-kernelspec uninstall venv_name`)

With the same directory open in cmd:

1. Install requirements from requirements.txt: <br>
`pip install -r requirements.txt`
2. Clone the [Matterport Mask_RCNN repository](https://github.com/matterport/Mask_RCNN): <br>
`git clone https://github.com/matterport/Mask_RCNN`
3. Change directory to Mask_RCNN in cmd using `cd Mask_RCNN`
4. Run `python setup.py install` to install the mask-rcnn module
5. Check installation using `pip show mask-rcnn`, expected output is:
```
Name: mask-rcnn
Version: 2.1
Summary: Mask R-CNN for object detection and instance segmentation
Home-page: https://github.com/matterport/Mask_RCNN
Author: Matterport
Author-email: waleed.abdulla@gmail.com
License: MIT
Location: path\monusac-instance-segmentation\venv_name\lib\site-packages\mask_rcnn-2.1-py3.7.egg
Requires:
Required-by:
```

### Dataset folder and code for test.ipynb referred from [here](https://github.com/jackfrost1411/MaskRCNN) (found from a towards data science article).