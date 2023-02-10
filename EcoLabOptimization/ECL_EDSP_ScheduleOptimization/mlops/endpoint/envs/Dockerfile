# FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
FROM mcr.microsoft.com/azureml/inference-base-1804:20230201.v1

# Create conda environment
RUN conda create -n minimal \
    python=3.9 \
    # pip=22.1.2 \
    # conda-forge::wheel==0.38.1 \
    # conda-forge::certifi==2022.12.07 \
    # conda-forge::setuptools==65.5.1 \
    -c anaconda -c conda-forge

ENV AZUREML_CONDA_ENVIRONMENT_PATH /opt/miniconda/envs/minimal

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

RUN pip install 'azureml-inference-server-http~=0.7.0'

ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH

COPY ./scheduleopt /scheduleopt
WORKDIR /scheduleopt
RUN echo $(python --version) && pip install -U pip && pip install setuptools && pip install -e .

CMD [ "runsvdir", "/var/runit" ]