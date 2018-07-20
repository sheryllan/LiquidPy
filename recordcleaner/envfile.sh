#!/bin/bash

if [ -z "${DIR}" ]; then
    export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

if [ -z "${OUTDIR}" ]; then
    export OUTDIR=${DIR}/results
fi

if [ -z "${VENV_PATH}" ]; then
    export VENV_PATH=${DIR}/venv

fi
export VENV_BIN=${VENV_PATH}/bin

export REQUIREMENTS=${DIR}/requirements.txt
export CMEGCHECK_PY=${DIR}/cmegcheck.py

export SHARED_FOLDER=//lcldn-fs1/shared
export SHARED_DES=DailyProdChecks
export USER=slan
export PASSWD=sheryl2002
export DOMAIN=liquid-capital

