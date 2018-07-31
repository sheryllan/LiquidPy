from google.cloud import datastore

from baseclasses import *
from settings import *


print(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

url = 'https://www.theice.com/marketdata/reports/datawarehouse/ConsolidatedEndOfDayReportPDF.shtml'
headers = {'User-Agent': USER_AGENT,
           'Accept': 'application/json'}
data = {'exchangeCode': 'IFES',
        'optionRequest': 'false',
        'reportId': 144,
        'selectionForm': '',
        'exchangeCodeAndContract': 'ALLCONTRACTS',
        'rcMode': 2}

rp = requests.post(url, json.dumps(data), headers=headers)

# Instantiates a client
datastore_client = datastore.Client()

# The kind for the new entity
kind = 'Task'
# The name/ID for the new entity
name = 'sampletask1'
# The Cloud Datastore key for the new entity
task_key = datastore_client.key(kind, name)

# Prepares the new entity
task = datastore.Entity(key=task_key)
task['description'] = 'Buy milk'

# Saves the entity
datastore_client.put(task)

print('Saved {}: {}'.format(task.key.name, task['description']))


class ICEScraper(ScraperBase):
    URL_ICE_FUT = 'https://www.theice.com/marketdata/reports/10'
    URL_ICE_OPT = 'https://www.theice.com/marketdata/reports/166'
    URL_ICE_S2F_FUT = 'https://www.theice.com/marketdata/reports/144'
    URL_ICE_S2F_OPT = 'https://www.theice.com/marketdata/reports/145'


    def scrape(self, report, rtime, **kwargs):
        pass