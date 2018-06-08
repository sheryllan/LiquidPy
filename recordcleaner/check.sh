#!/bin/bash

export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export VENV_NAME=venv
export VENV_PATH = ${DIR}/${VENV_NAME}
export VENV_BIN=${VENV_PATH}/bin
export OUTDIR=${DIR}/results

readArgs()

if [ "${CLEAN}" = true ] || [ -d ${VENV_PATH} ]; then
	setupEnv()
if

source ${VENV_BIN}/activate
echo -e "\nRunning cmegcheck.py"
python cmegcheck.py -d "${OUTDIR}"
deactivate


readArgs() {
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
	
	
setupEnv() {
	rm -rf ${VENV_PATH}

	mkdir ${VENV_PATH}
	scl enable rh-python36 "virtualenv ${VENV_NAME}"
	echo "Virutal environment ${VENV_NAME} created"

	source ${VENV_BIN}/activate
	echo "virtual environment ${VENV_NAME} activated"

	echo "Installing missing packages from requirements.txt"
	pip install -r requirements.txt

	echo "Packages installed in ${VENV_NAME}:"
	pip freeze
	
	deactivate
