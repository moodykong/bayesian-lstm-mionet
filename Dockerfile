FROM python:3.10-bookworm

RUN apt update && apt -y upgrade 

# Create the working directory in the container.
WORKDIR /LSTM-MIONet

# Azure CLI
# RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Copy the requirements file located here and copy it into the container's working directory.
COPY ./requirements.txt ./

# Pip Dependencies
RUN pip install --upgrade pip 
RUN pip install -r ./requirements.txt

# Azure CLI ML Extension
# RUN az extension add -n ml