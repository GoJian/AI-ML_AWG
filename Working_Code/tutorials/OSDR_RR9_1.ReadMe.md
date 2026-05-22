### Install and import modules in Google Colab
```
!pip install numpy
!pip install pandas
!pip install -U -q PyDrive
!pip install scanpy

!pip install gseapy
!pip install pydeseq2
!pip install pybiomart
!pip install mygene
```

### Local python environment SETUP

In local environment, we can follow the steps below to set up your python environment
1. Download and setup a conda environment using [miniforge](https://github.com/conda-forge/miniforge#miniforge3). 
   Then, activate your conda environment at the command line terminal with: 

   ```
   conda activate
   ```
2. Using git to clone this github repository:

   ```
   git clone git@github.com:GoJian/AI-ML_AWG.git
   ```

3. Go to the folder "AI-ML_AWG" and create a conda environment by run this line in terminal. This can take a while
   ```
   conda env create -f environment.yml
   ```

   Activate the conda environment with:
   ```
   conda activate awg
   ```

4. Run these additional commands to install additonal packages using pip:
   ```
   pip install gseapy
   pip install pydeseq2
   pip install pybiomart
   pip install mygene
   ```