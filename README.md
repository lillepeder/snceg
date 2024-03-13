# snceg - Substantia Nigra pars compacta Segmentation with Attention U-Net.
Attention U-Net trained on multi-modal MRI to segment the substantia nigra pars compacta.

The model is used to derive SNc masks from neuromelanin-weighted MRI, but is relatively robust to other MRI modalities.

The model is Pytorch-based and implemented with [fastMONAI](https://github.com/MMIV-ML/fastMONAI/tree/master). For general tutorials on building models with fastMONAI, check their [documentation](https://fastmonai.no/).
[Model repository](https://huggingface.co/lillepeder/SNceg-0.1)


## Installation
The package dependencies for running the notebooks can be installed by running:

_Optional_ conda environment<br>
`conda create --name snceg python=3.11`<br>
`activate snceg`

`pip install nibabel nilearn huggingface_hub fastMONAI ipykernel`  

[FreeSurfer's](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) `freeview` should be installed for interactive visualization 

## Contents
- nbs/prediction.ipynb : Curated notebook for downloading and applying our pretrained model
- data/mean_NM.nii.gz : Example dataset produced by averaging NM weighted MRIs from 82 subjects


## Disclaimer
The model is merely a research tool. It has only been tested on gradient echo (GRE) images and performance will depend on the imaging protocol. 
We encourage others to test it on their own datasets, and judge for themselves. The model may also be finetuned to your own data as the model parameters are [openly available](https://huggingface.co/lillepeder/SNceg-0.1).