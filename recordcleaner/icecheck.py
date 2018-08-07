from google.cloud import datastore
from time import sleep, time
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


from baseclasses import *
from settings import *

#
# driver = webdriver.PhantomJS(PHANTOMJS)
# # options = Options()
# # options.headless = True
# # driver = webdriver.Firefox(executable_path=GECKODRIVER, firefox_options=options)
# driver.get('https://www.google.com/recaptcha/api2/webworker.js?h1=en&v=v1531759913576')
# pre = driver.find_element_by_tag_name('pre')
# print(pre.text)

# url = 'https://www.theice.com/marketdata/reports/datawarehouse/ConsolidatedEndOfDayReportPDF.shtml'
# headers = {'User-Agent': USER_AGENT,
#            'Accept': 'application/json'}
# data = {'exchangeCode': 'IFES',
#         'optionRequest': 'false',
#         'reportId': 144,
#         'selectionForm': '',
#         'exchangeCodeAndContract': 'ALLCONTRACTS',
#         'rcMode': 2}
#
# rp = requests.post(url, json.dumps(data), headers=headers)
#
# # Instantiates a client
# datastore_client = datastore.Client()
#
# # The kind for the new entity
# kind = 'Task'
# # The name/ID for the new entity
# name = 'sampletask1'
# # The Cloud Datastore key for the new entity
# task_key = datastore_client.key(kind, name)
#
# # Prepares the new entity
# task = datastore.Entity(key=task_key)
# task['description'] = 'Buy milk'
#
# # Saves the entity
# datastore_client.put(task)
#
# print('Saved {}: {}'.format(task.key.name, task['description']))
#

class ICEScraper(ScraperBase):
    URL_ICE_FUT = 'https://www.theice.com/marketdata/reports/10'
    URL_ICE_OPT = 'https://www.theice.com/marketdata/reports/166'
    URL_ICE_S2F_FUT = 'https://www.theice.com/marketdata/reports/144'
    URL_ICE_S2F_OPT = 'https://www.theice.com/marketdata/reports/145'

    def __init__(self):
        super().__init__()

    def get_eod_report(self, url):
        # driver = webdriver.PhantomJS(PHANTOMJS)
        options = Options()
        options.headless = True
        binary = FirefoxBinary(FIREFOX)
        driver = webdriver.Firefox(firefox_binary=binary, executable_path=GECKODRIVER)

        # options = webdriver.ChromeOptions()
        # options.add_argument('--ignore-certificate-errors')
        # options.add_argument('--test-type')
        # options.binary_location = CHROME
        # driver = webdriver.Chrome(chrome_options=options, executable_path=CHROMEDRIVER)
        driver.get(url)
        # wait = WebDriverWait(driver, 30)

        mainWin = driver.current_window_handle

        driver.switch_to.frame(driver.find_elements_by_tag_name("iframe")[0])

        check_box = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "recaptcha-anchor"))
        )

        sleep(3)
        check_box.click()

        driver.find_element_by_class_name('btn').click()
        driver.switch_to.window(mainWin)

        print(driver)




    def scrape(self, report, rtime, **kwargs):
        pass


if __name__ == '__main__':
    s = ICEScraper()
    s.get_eod_report(s.URL_ICE_FUT)
