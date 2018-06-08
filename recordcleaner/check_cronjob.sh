#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

. ${DIR}/check.sh

SHARED_FOLDER=//lcldn-fs1/shared
SHARED_DES=DailyProdChecks
USER=slan
PASSWD=sheryl2002
DOMAIN=liquid-capital

cd ${OUTDIR}
smbclient ${SHARED_FOLDER} ${PASSWD} -W ${DOMAIN} -U ${USER} -c "prompt OFF; cd ${SHARED_DES}; mput *.xlsx"
echo "Results copied to shared folder: ${SHARED_FOLDER}/${SHARED_DES}"

atts=""
echo "Sending email with attachments:"
for fp in ./*.xlsx; do
	echo ${fp}
	atts="${atts} -a ${fp}"
done

TITLE=" -s \"Daily Check Results\""
RECIPIENT=" slan@liquidcapital.com"
MAILCMD="mail${TITLE}${atts}${RECIPIENT}"
MAILINFO="echo Check result sent" 

eval ${MAILINFO} "|" ${MAILCMD}
