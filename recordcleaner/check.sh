#!/bin/bash


report_time() {
    fmt=${1:-1}
    if [ "${fmt}" = 1 ]; then
        echo $( date -d "$(date +%Y-%m-15) -1 month" +%Y%m )
    elif [ "${fmt}" = 0 ]; then
        echo $( date -d "$(date +%Y-%m-15) -1 month" )
    else
        echo $( date -d "$(date +%Y-%m-15) -1 month" +"${fmt}" )
    fi
}


check_outpath() {
    echo "${OUTDIR}/$1_$( report_time )_checked.xlsx"
}

match_outpath() {
    echo "${OUTDIR}/$1_$( report_time )_all.xlsx"
}

python_sript() {
    echo "${DIR}/${1,,}check.py"
}

EXCHANGES=(CMEG EUREX OSE)
RTIME="$(report_time "%Y %m")"


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
	--exch)
	    EXCHANGES=($2)
	    shift
	    shift
	;;
	--rtime)
	    RTIME=($2)
	    shift
	    shift

esac
done

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIR}/env_setup.sh ${CLEAN} --keep

export CHECK_OUTFILES=()
export MATCH_OUTFILES=()
for exch in ${EXCHANGES[@]}; do
    echo -e "\nRunning $( python_sript ${exch} )"
    coutpath="$( check_outpath ${exch^^} )"
    soutpath="$( match_outpath ${exch^^} )"
    CHECK_OUTFILES+=("${coutpath}")
    MATCH_OUTFILES+=("${soutpath}")
    python "$( python_sript ${exch} )" ${ICINGA} ${LOGLEVEL} --coutpath "${coutpath}" --soutpath "${soutpath}" --rtime ${RTIME}
done

deactivate
