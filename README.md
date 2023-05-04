# 11785-final-project

### Set up the environment
To install the Anaconda virtual environment that we used to develop this project, run the commands below.
```
conda env create -f environment.yml
conda activate ras
```

### Data preparation
There are two steps of data preparation in our project. First, we randomly select two other audio clips for each audio sample in the FAVD dataset and concatenate the three audio clips. This results in a new dataset where each sample contains both positive and negative segments corresponding to the caption.
1. Run the jupyter notebook file combine_data.ipynb, the output train.json, val.json and test.json should be put under ./, these are the files for training, validation and testing.
2. Run the jupyter notebook file data_prep.ipynb, this notebook process the json files we get in 1, calculate and store the audio feature computed by pre-trained wav2vec2 model for all the samples. Note that we concatenate the audio clips according to the json files and does not store the concatenated audio. To run this notebook, the original data should be put under train/, val/ and test/, respectively. 

### Train the model
Run the following command to train the model from scratch.
```
python train.py
```

### Inference
Run the code in inference.ipynb to do inference.

### Demo Files
The input and output demo files are under the directory demo/

### Download the link
http://www.avlbench.opennlplab.cn/download

The user need to submit an application via the Google form provided in the link above for downloading the dataset.
