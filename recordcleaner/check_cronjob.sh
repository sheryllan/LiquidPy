#!/bin/bash

logit() {
    while read
    do
        echo "$( date +'%Y-%m-%d %T' ) $REPLY" >> ${LOG_FILE}
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

LOG_PATH="$( dirname "$( pwd )" )/log"

if ! [ -d ${LOG_PATH} ]; then
    mkdir ${LOG_PATH}
fi

LOG_FILE="${LOG_PATH}/check_cron_$( date +"%Y%m%d" ).log"
exec 3>&1 1>> >(logit) 2>&1
echo "Logging to ${LOG_FILE}"
source check.sh --icinga --loglevel INFO

smbclient ${SHARED_FOLDER} ${PASSWD} -W ${DOMAIN} -U ${USER} -c "prompt OFF; lcd ${OUTDIR}; cd ${SHARED_DES}; mput *$( report_time )*.xlsx"
echo "Results copied to shared folder: ${SHARED_FOLDER}/${SHARED_DES}"

echo "Sending email with attachments:"
for fp in $( found "${CHECK_OUTFILES[@]}" ); do
    echo ${fp}
    ATTACHMENTS+="-a ${fp} "
done

TITLE="-s \"$( date -d $( report_time )'01' +'%b %Y' ) Monthly Check Results\""
RECIPIENT="slan@liquidcapital.com,alpha@liquidcapital.com,jzhou@liquidcapital.com,jsu@liquidcapital.com"

MAILCMD="mail ${TITLE} ${ATTACHMENTS} ${RECIPIENT}"
MAILINFO=$( echo "Missing files:"; not_found "${CHECK_OUTFILES[@]}" )

eval "echo '${MAILINFO}' | ${MAILCMD}"
