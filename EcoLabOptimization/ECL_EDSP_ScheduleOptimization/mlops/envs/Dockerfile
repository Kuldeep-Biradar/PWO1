FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
COPY ./scheduleopt /scheduleopt
WORKDIR /scheduleopt
RUN echo $(python --version) && pip install -U pip && pip install setuptools && pip install -e .