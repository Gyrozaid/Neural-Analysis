# Tetrode Processing

![alt text](image-1.png)

This repo houses the processing pipeline for electrode data visualization.


## Setup

To get started, first we will set up the necessary conda environments. Make sure you have anaconda installed.

### SpikeSorting

To set up the spike sorting environment:

Within the anaconda prompt:
1. run ```shell conda create --name SpikeSorting python=3.9```
2. run ```shell conda activate SpikeSorting```
3. run ```shell python -m pip install kilosort[gui] mountainsort5```
4. run ```shell pip install spikeinterface```

### Phy 2

To set up the visualization environment:

Within the anaconda prompt:

1. run ```shell conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets```
2. run ```shell conda activate phy2```
3. run ```shell pip install git+https://github.com/cortex-lab/phy.git```
4. run ```shell pip install klusta klustakwik2```

## Usage

1. navigate to the project folder within your anaconda prompt
2. run ```shell conda activate SpikeSorting```
2. run ```shell python sort.py -p "enter path to your data" -s "enter path to your saved results" -a "enter name of sorting algorithm"```
    - options for sorting algorithms currently include "kilosort4", "spykingcircus2", and "mountainsort5"
3. run ```shell conda activate phy2```
4. run ```shell phy template-gui path_to_your_phy_python_file```
    - sort.py will automatically export to phy and provide the command to open results with phy for you. If you do not see this command, run the above command replacing path_to_your_phy_python_file with the params.py file you see within your saved results 

Have fun spike sorting!




MAIN FILE: 
sort.ipynb
comment out processing steps that have already been computed.
