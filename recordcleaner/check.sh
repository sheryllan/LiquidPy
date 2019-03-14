#!/bin/bash

while [ $# -gt 0 ]
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

report_time() {
    echo $( date -d "$(date +%Y-%m-15) -1 month" +%Y%m )
}

check_outpath() {
    echo "${OUTDIR}/$1_$( report_time )_checked.xlsx"
}

match_outpath() {
    echo "${OUTDIR}/$1_$( report_time )_all.xlsx"
}


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIR}/env_setup.sh ${CLEAN}

source ${VENV_BIN}/activate

echo -e "\nRunning ${CMEGCHECK_PY}"
CMEG_OUTFILES=($( check_outpath CMEG ) $( match_outpath CMEG ))
python ${CMEGCHECK_PY} ${ICINGA} ${LOGLEVEL} --coutpath ${CMEG_OUTFILES[0]} --soutpath ${CMEG_OUTFILES[1]}

echo -e "\nRunning ${OSECHECK_PY}"
OSE_OUTFILES=($( check_outpath OSE ) $( match_outpath OSE ))
python ${OSECHECK_PY} ${ICINGA} ${LOGLEVEL} --coutpath ${OSE_OUTFILES[0]} --soutpath ${OSE_OUTFILES[1]}

echo -e "\nRunning ${EUREXCHECK_PY}"
EUREX_OUTFILES=($( check_outpath EUREX ) $( match_outpath EUREX ))
python ${EUREXCHECK_PY} ${ICINGA} ${LOGLEVEL} --coutpath ${EUREX_OUTFILES[0]} --soutpath ${EUREX_OUTFILES[1]}

deactivate

export CHECK_OUTFILES=(${CMEG_OUTFILES[0]} ${OSE_OUTFILES[0]} ${EUREX_OUTFILES[0]})
export MATCH_OUTFILES=(${CMEG_OUTFILES[1]} ${OSE_OUTFILES[1]} ${EUREX_OUTFILES[1]})