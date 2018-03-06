import urllib2
import os
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTRect, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7; X11; Linux x86_64) ' \
             'Gecko/2009021910 Firefox/3.0.7 Chrome/23.0.1271.64 Safari/537.11'


def download(url, filename):
    request = urllib2.Request(url, headers={'User-Agent': user_agent})

    try:
        response = urllib2.urlopen(request)
        with open(filename, 'wb') as fh:
            print('\n[*] Downloading: {}'.format(os.path.basename(filename)))
            fh.write(response.read())
            print ('\n[*] Successful')
    except urllib2.HTTPError, e:
        print e.fp.read()


class PDFHelper(object):
    @staticmethod
    def get_text_from_ltobj(obj):
        if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine):
            return obj.get_text()
        else:
            raise ValueError('No text found in the given LTObject')

    @staticmethod
    def extract_pdf_sections(doc, levels):
        return [(level, title) for (level, title, dest, a, structelem) in doc.get_outlines()
                if level in levels]

    @staticmethod
    def setup_pdfdocument(fh):
        parser = PDFParser(fh)
        doc = PDFDocument(parser)
        return doc

    @staticmethod
    def setup_interpreter():
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        return PDFPageInterpreter(rsrcmgr, device), device


class CME(object):
    url_adv = 'http://www.cmegroup.com/daily_bulletin/monthly_volume/Web_ADV_Report_CMEG.pdf'
    pdf_adv = 'CME_Average_Daily_Volume.pdf'

    url_prodslate = 'http://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.xls'
    xls_prodslate = 'Product_Slate.xls'

    # metadata in adv.pdf
    adv_pdf_levels = [3, 4]
    adv_date_no = 1
    adv_headers_no = range(3, 11)

    def __init__(self, download_path):
        self.download_path = download_path
        self.full_path_adv = os.path.join(self.download_path, self.pdf_adv)
        self.full_path_prodslate = os.path.join(self.download_path, self.xls_prodslate)

    def download_adv(self):
        download(self.url_adv, self.full_path_pdf_adv)

    def download_prodslate(self):
        download(self.url_prodslate, self.full_path_prodslate)

    def parse_pdf_adv(self):
        with open(self.full_path_adv, 'rb') as infile:
            doc = PDFHelper.setup_pdfdocument(infile)
            sections = PDFHelper.extract_pdf_sections(doc, self.adv_pdf_levels)
            interpreter, device = PDFHelper.setup_interpreter()

            groups = self.get_product_group(sections)
            for i, page in enumerate(PDFPage.create_pages(document=doc)):
                interpreter.process_page(page)
                ltobjs = list(device.get_result())

                if i == 0:
                    date = self.get_adv_todate(ltobjs)
                    headers = self.get_adv_headers(ltobjs)

                for obj in ltobjs:
                    pdftext = PDFHelper.get_text_from_ltobj(obj)




    def get_product_group(self, sections):
        prev_level = sections[0][0]
        asset = sections[0][1]
        result = dict()
        for level, title in sections[1:]:
            if level < prev_level:
                asset = title
            else:
                instrument = title
                result[('{} {}'.format(asset, instrument))] = (asset, instrument)
            prev_level = level
        return result

    def get_adv_headers(self, ltobjs):
        return [(' '.join(PDFHelper.get_text_from_ltobj(ltobjs[i]).split('\n'))).rstrip()
                for i in self.adv_headers_no]

    def get_adv_todate(self, ltobjs):
        return PDFHelper.get_text_from_ltobj(ltobjs[self.adv_date_no]).rstrip()




cme = CME('/home/slan/Documents/downloads/')
# cme.download_prodslate()
tb = cme.parse_pdf_adv()
