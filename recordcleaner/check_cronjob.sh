#!/bin/bash

logit() {
    while read
    do
        echo "$( date +'%Y-%m-%d %T' ) $REPLY"  >> ${LOG_FILE}
    done
}

cd "$( dirname "${BASH_SOURCE[0]}" )"

LOG_FILE="log/check_cron_$( date +"%Y%m%d" ).log"
exec 3>&1 1>> >(logit) 2>&1

source ./check.sh --icinga

smbclient ${SHARED_FOLDER} ${PASSWD} -W ${DOMAIN} -U ${USER} -c "prompt OFF; lcd ${OUTDIR}; cd ${SHARED_DES}; mput *.xlsx"
echo "Results copied to shared folder: ${SHARED_FOLDER}/${SHARED_DES}"

ATTS=""
echo "Sending email with attachments:"
for fp in ${OUTDIR}/*.xlsx; do
	echo ${fp}
	ATTS="${ATTS} -a ${fp}"
done

TODAY="$(date +'%d/%m/%Y')"
TITLE=" -s \"Daily Check Results - ${TODAY}\""
RECIPIENT=" slan@liquidcapital.com"
MAILCMD="mail${TITLE}${ATTS}${RECIPIENT}"
MAILINFO="echo Check result sent" 

eval ${MAILINFO} "|" ${MAILCMD}
