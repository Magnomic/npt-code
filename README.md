# Description

The implemention of NPT, *Disentangling Noise Pattern from Seismic Images: Noise Reduction and Style Transfer* . 

NPT has two key parts, which are named I2I-NT and D2D-NT. I2I-NT provides image to image noise transfer while D2D-NT trains dataset to dataset noise transfer using the same network structure with DnCNN.

For new readers, we recommend they follow the steps below to better understand our model:

* 1. Build conda environment using requirments.txt
* 2. Read & run i2i_nt.ipynb to see how I2I-NT works
* 3. Read d2d_nt.py, then checking its configuration in /options
* 4. Run & check comparation.ipynb to see the results

----

The directory structure and files of this project is detailed as:

|Directory | Description |
|----|----|
|/data | Seismic data processing method for D2D-NT training |
|/fault_interpretation | Experiments for transferability|
|/model_zoo | Npt models that are trained by us, which outputs results in paper.|
|/models | Network structures of D2D-NT and baselines|
|/options | Hyper-parameter and configuration of D2D-NT model|
|/parameter_test | Data patches that used in our experiments|
|/tdtv_patches| Samples that ouputs by TDTV|
|/utils | Models that used for seismic image processing|
|comparation.ipynb | Experiment results in our paper|
|d2d_nt.py | Code of D2D-NT model|
|i2i_nt.ipynb | Code of I2I-NT model. We also provide some examples for readers to fine-tune the parameters on their datasets.|
|image_utils.py | Utils that used for experiments|
|no_clean.png | Position image|
|no_noise.png | Position image|
|requirements.txt | Readers can use this file to build a conda runtime environment|
|ricker.ipynb | Method to generate our FSSynth dataset.|
|tdtv.py | TDTV model that implemented by us|
|tdtv_validate | The smoothing process implemention and examples of TDTV model|
