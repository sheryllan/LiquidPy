#!/bin/bash

if [ -z "${DIR}" ]; then
	export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

ENVFILE=${DIR}/envfile.sh
echo "Sourcing all the environment variables from ${ENVFILE}"
source ${ENVFILE}

if [ -z "${VENV_PATH}" ]; then
	echo "ERROR: VENV_PATH variable unset"
	exit 1
fi

(
while [[ $# -gt 0 ]]
do
	key="$1"
case $key in
	--clean)
		CLEAN=true
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
)

source ${VENV_BIN}/activate
echo "Virtual environment ${VENV_PATH} activated"

echo "Installing missing packages from requirements.txt"
pip install -r ${REQUIREMENTS}

echo "Packages installed in ${VENV_PATH}:"
pip freeze

deactivate
