#!/bin/bash

logit() {
    while read
    do
        echo "$( date +'%Y-%m-%d %T' ) $REPLY" | tee -a ${LOG_FILE}
    done
}

found() {
    for f in "$@"; do
        ! [ -f "${f}" ] || echo ${f}
    done
}

not_found() {
    for f in "$@"; do
        [ -f "${f}" ] || echo ${f}
    done
}

cd "$( dirname "${BASH_SOURCE[0]}" )"
OUTDIR="$( dirname "$( pwd )" )/results"
LOG_PATH="$( dirname "$( pwd )" )/log"

if ! [ -d ${LOG_PATH} ]; then
    mkdir ${LOG_PATH}
fi

LOG_FILE="${LOG_PATH}/check_cron_$( date +"%Y%m%d" ).log"
exec 3>&1 4>&2 &> >(logit)
echo "Logging to ${LOG_FILE}"


ENV=prod
while [ $# -gt 0 ]
do
	key="$1"
case $key in
    --env)
    ENV=$2
    shift
    shift
    ;;
esac
done

if [ "${ENV}" = prod ]; then
    RECIPIENT="slan@liquidcapital.com,alpha@liquidcapital.com,jzhou@liquidcapital.com,jsu@liquidcapital.com"
elif [ "${ENV}" = dev ]; then
    RECIPIENT="slan@liquidcapital.com"
fi

source check.sh --icinga --loglevel INFO --outdir "${OUTDIR}"


SHARED_FOLDER=//lcldn-fs1/shared
SHARED_DES=DailyProdChecks/
USER=slan
PASSWD=sheryl2002
DOMAIN=liquid-capital

smbclient ${SHARED_FOLDER} ${PASSWD} -W ${DOMAIN} -U ${USER} -c "prompt OFF; lcd ${OUTDIR}; cd ${SHARED_DES}; mput *$( report_time )*.xlsx"
echo "Results copied to shared folder: ${SHARED_FOLDER}/${SHARED_DES}"

FOUND=$( found "${CHECK_OUTFILES[@]}" )
NOT_FOUND=$( not_found "${CHECK_OUTFILES[@]}" )
ATTACHMENTS=$( echo "$FOUND"  | sed "s/^/-a /" | tr "\n" " " )

MAILINFO="Check results attached."
if [ -n "${NOT_FOUND}" ]; then
    MISSING=$( echo "Missing files:"; echo "${NOT_FOUND}" )
    MAILINFO=$( echo "${MAILINFO}"; echo "${MISSING}")
fi

TITLE="-s '$( date -d $( report_time )'01' +'%b %Y' ) Monthly ADV Check Results'"
eval "echo ${MAILINFO} | mail ${TITLE} ${ATTACHMENTS} ${RECIPIENT}"

if [ -n "${FOUND}" ]; then
    echo "Email sent to ${RECIPIENT} with attachments:"
    echo "${FOUND}"
else
    echo "Email sent to ${RECIPIENT} without any attachments"
fi
echo ${MISSING}

exec 1>&3 2>&4