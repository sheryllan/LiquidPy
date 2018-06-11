#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
echo "running under current directory $PWD"

VENV_NAME=venv
if [ -d ${VENV_NAME} ];then 
	rm -rf ${VENV_NAME}
	echo "deleted the existing virual environment ${VENV_NAME}"
fi

mkdir ${VENV_NAME}
scl enable rh-python36 "virtualenv ${VENV_NAME}"
echo "virutal environment ${VENV_NAME} created"

source ${VENV_NAME}/bin/activate
echo "virtual environment ${VENV_NAME} activated"

echo "Installing missing packages from requirements.txt"
pip install -r requirements.txt

echo "packages installed in ${VENV_NAME}:"
pip freeze

echo -e "\nrunning cmegcheck.py"
python cmegcheck.py

deactivate


