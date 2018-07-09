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

export ICINGA_HOST=lcldn-icinga1
export ICINGA_API_PORT=5665
export ICINGA_API_PCR=v1/actions/process-check-result
export ICINGA_API_USER=icinga
export ICINGA_API_PSW=icinga2002
export CA_CRT=ca.crt

export SVC_CME=cme_check
export SVC_CBOT=cbot_check
export SVC_NYMEX=nymex_check
export SVC_OSE=ose_check

export SHARED_FOLDER=//lcldn-fs1/shared
export SHARED_DES=DailyProdChecks
export USER=slan
export PASSWD=sheryl2002
export DOMAIN=liquid-capital

