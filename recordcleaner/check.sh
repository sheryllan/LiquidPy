#!/bin/bash

while [[ $# -gt 0 ]]
do
	key="$1"
case $key in
	--clean)
		CLEAN=--clean
		shift
	;;
	--icinga)
		ICINGA=--icinga
		shift
	;;
	--loglevel)
	    LOGLEVEL="--loglevel $2"
	    shift
	    shift
	;;
	--outdir)
	    OUTDIR=$2
	    shift
	    shift
	;;

esac
done

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIR}/env_setup.sh ${CLEAN}

source ${VENV_BIN}/activate
echo -e "\nRunning ${CMEGCHECK_PY}"
python ${CMEGCHECK_PY} ${ICINGA} ${LOGLEVEL} --coutpath --soutpath

deactivate
