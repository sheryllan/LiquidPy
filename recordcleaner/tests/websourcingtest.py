import unittest as ut

from commonlib.websourcing import *
import json


class MainTests(ut.TestCase):
    def test_http_post(self):
        url = 'https://lcldn-icinga1:5665/v1/actions/process-check-result'
        data = {"filter": "service.name==\"cme_check\"",
                "type": "Service",
                "exit_status": 2,
                "plugin_output": "PING CRITICAL - Packet loss = 100%",
                "check_source": "linuxdev119"}
        data = json.dumps(data)
        cert = '/usr/local/share/ca-certificates/ca.crt'
        auth = ('icinga', 'icinga2002')

        http_post(url, data, auth, cert)

