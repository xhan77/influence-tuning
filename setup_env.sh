######## Environment on local machine ########

# setup conda first
conda create --name gif_pt160cu101 python=3.7

# activate env (might need to change to source from conda?)
source activate gif_pt160cu101

# install pytorch and cudatoolkit
conda install pytorch=1.6.0 cudatoolkit=10.1 -c pytorch

# install bert related packages
pip install pytorch_pretrained_bert==0.6.1 transformers==4.2.0 datasets==1.2.1 pyyaml==5.4.1

# # Not using horovod anymore

# # prepare for horovod installation
# pip install cmake==3.18.4
# conda install gcc_linux-64 gxx_linux-64
# conda install -c conda-forge openmpi=4

# make sure you have CUDA in bash path
# e.g., "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/lib64" and "export PATH=${PATH}:/usr/local/cuda-10.1/bin" in ~/.bashrc
# need to check /usr/local/cuda-10.1/include/x86... (cublas_v2.h)

# # install horovod
# HOROVOD_CUDA_HOME=/usr/local/cuda-10.1/ HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]

######## Environment on cluster ########

# setup conda first
conda create --name gif_cluster python=3.7

# activate env (might need to change to source from conda?)
source activate gif_cluster

# install pytorch and cudatoolkit
conda install pytorch=1.7.1 cudatoolkit=11.0 -c pytorch

# install bert related packages
pip install pytorch_pretrained_bert==0.6.1 transformers==4.2.0 datasets==1.2.1 pyyaml==5.4.1

# # Not using horovod anymore

# # prepare for horovod installation
# pip install cmake==3.18.4
# conda install gcc_linux-64 gxx_linux-64
# conda install -c conda-forge openmpi=4

# make sure you have CUDA in bash path
# e.g., "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.0/lib64" and "export PATH=${PATH}:/usr/local/cuda-11.0/bin" in ~/.bashrc
# need to check /usr/local/cuda-11.0/include/x86... (cublas_v2.h)

# # install horovod
# HOROVOD_CUDA_HOME=/usr/local/cuda-11.0/ HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[pytorch]
