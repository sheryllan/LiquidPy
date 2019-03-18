#!/bin/bash

if [ -z "${DIR}" ]; then
	export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

export VENV_PATH=${DIR}/venv


while [[ $# -gt 0 ]]
do
	key="$1"
case $key in
	--clean)
		CLEAN=true
		shift
	;;
	--keep)
		KEEP=true
		shift
	;;
esac
done

if [ "${CLEAN}" = true ] || ! [ -d ${VENV_PATH} ]; then
	echo "Making the virtual environment directory: ${VENV_PATH}"
	rm -rf ${VENV_PATH}
	mkdir ${VENV_PATH}
	
	scl enable rh-python36 "virtualenv ${VENV_PATH}"
	echo "Virutal environment ${VENV_PATH} created"
fi


VENV_BIN=${VENV_PATH}/bin
REQUIREMENTS=${DIR}/requirements.txt

source ${VENV_BIN}/activate
echo "Virtual environment ${VENV_PATH} activated"

echo "Installing missing packages from requirements.txt"
pip install -r ${REQUIREMENTS}

echo "Packages installed in ${VENV_PATH}:"
pip freeze

if [ "${KEEP}" != true ]; then
    deactivate
fi
