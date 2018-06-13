#!/usr/bin/env bash

if [ -z "${DIR}" ]; then
export DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

export VENV_NAME=venv
export VENV_PATH=${DIR}/${VENV_NAME}
export VENV_BIN=${VENV_PATH}/bin
export OUTDIR=${DIR}/results


SHARED_FOLDER=//lcldn-fs1/shared
SHARED_DES=DailyProdChecks
USER=slan
PASSWD=sheryl2002
DOMAIN=liquid-capital
