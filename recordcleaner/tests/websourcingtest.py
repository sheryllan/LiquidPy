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
        print(json.dumps(data, indent=4))
        data = json.dumps(data)
        cert = '/usr/local/share/ca-certificates/ca.crt'
        auth = ('icinga', 'icinga2002')

        http_post(url, data, auth, cert)



    def test_json(self):
        import os
        os.environ.setdefault('DIR', '/home/slan/Python/recordcleaner')
        from baseclasses import IcingaHelper, TaskBase
        from settings import CMEGSetting

        jd = IcingaHelper.to_json(True, 'cme_check')
        tb = TaskBase(CMEGSetting)

        r = tb.post_pcr(jd)
        print()


