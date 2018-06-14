#!/usr/bin/env bash

if [ -z "${DIR}" ]; then
export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

export VENV_NAME=venv
export VENV_PATH=${DIR}/${VENV_NAME}
export VENV_BIN=${VENV_PATH}/bin
export OUTDIR=${DIR}/results
export ICINGA_HOST=lcldn-icinga1
export ICINGA_API_PORT=5665
export ICINGA_PCR=https://${ICINGA_HOST}:${ICINGA_API_PORT}/v1/actions/process-check-result

export SVC_CME=cme_check
export SVC_OSE=ose_check

SHARED_FOLDER=//lcldn-fs1/shared
SHARED_DES=DailyProdChecks
USER=slan
PASSWD=sheryl2002
DOMAIN=liquid-capital
