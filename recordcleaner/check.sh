#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"
ENVFILE=envfile.sh
source ${ENVFILE}

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
}
	
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
}


readArgs
if [ "${CLEAN}" = true ] || ! [ -d ${VENV_PATH} ]; then
    setupEnv
fi

source ${VENV_BIN}/activate
echo -e "\nRunning cmegcheck.py"
python cmegcheck.py
deactivate
