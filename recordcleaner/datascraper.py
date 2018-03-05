import urllib2
import os


def download(url, filename):
    response = urllib2.urlopen(url)
    with open(filename, 'wb') as fh:
        print('\n[*] Downloading: {}'.format(os.path.basename(filename)))
        fh.write(response.read())
        print ('\n[*] Successful')



class CME(object):
    url_adv = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_Volume_Report_CMEG.pdf'
    pdf_adv = 'CME_Average_Daily_Volume.pdf'

    def __init__(self, dlpath):
        download_path = dlpath



cme = CME('/home/slan/Documents/downloads/')
download(cme.url_adv, os.path.join(cme.download_path, cme.pdf_adv))
