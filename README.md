# HDC 2021 - TU Berlin & 



In order to use our trained deblurring model, the weights need to be downloaded first.
The weights are available here (Link) and should be stored in the folder “weights” without changing the folder structure of the downloaded file. 
If you wish to store the weights at another location, please specify the (relative or absolute) path to that location in the config.py file by changing the variable WEIGHTS_PATH.

In order to avoid unexpected behaviour due to diverging library and framework versions, we have included the yaml file hdc_env.yml from which our conda environment can be reconstructed using the command: conda env create -f hdc_env.yml
The environment is called hdc and can be started with conda activate hdc

Once all that is done, the main.py routine can be called from the command line. It expects three arguments as specified in the HDC Rules (input folder, output folder, step). 
main.py will run on the GPU if one is available and on the CPU otherwise.
