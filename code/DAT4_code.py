# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:34:55 2015

@author: Jason
"""

import csv
import os
import pycrunchbase
import re
import requests
import sys
import time
import urllib
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    
sns.set_style('white')    

# Download and extract patent zip files
#===================================================================    
# Download the USPTO bulk download webpage 
r = requests.get\
("https://www.google.com/googlebooks/uspto-patents-grants-text.html")

# Set the regex pattern for the files.
# The format of the filenames is 'YYYY/ipgYYMMDD.zip', e.g. 2014/ipg140318.zip. 
# Filenames from 2004 and before omit the initial 'i', but that ended up being irrelevant.
# Regex pattern = \d+/i?pg\d+\.zip
re_tag = re.compile\
("http://storage\.googleapis\.com/patents/grant_full_text/\d+/i?pg\d+\.zip")

# Create a list of the relevant urls
zips = re.findall(re_tag, r.text)

zip_destination_path = '../data/USPTO_zipfiles/'

# Download and extract the zipfiles 
for url in zips:
    filename =  url[59:]
    try:
        urllib.urlretrieve(url, filename)
    except ValueError:
        continue
    sourceZip = zipfile.ZipFile(filename, 'r')
        
    for name in sourceZip.namelist():
        sourceZip.extract(name, zip_destination_path)
    sourceZip.close

# Reformat xmls
#===================================================================  
xml_destination_path = "../data/formatted_xmls/"

xml_source_list = [zip_destination_path + x for x \
in os.listdir(zip_destination_path)[1:]]

# Reformat xmls to work with XML parser
for xml in xml_source_list:
    text = '<patents>\n'
    with open(xml,'rU') as f:
       for line in f.readlines():
           if '<?xml' not in line and '<!DOCTYPE' not in line: 
               text += line
    text += '</patents>\n'
    with open(xml_destination_path + xml[-13:], 'wb') as g:
        g.write(text)

# Process xmls and output relevant data to csv
#===================================================================          
csv_destination_path = "../data/CSVs/"

csv_source_list = [xml_destination_path + x for x \
in os.listdir(xml_destination_path)[1:]]

# Text fields in patents can be very large, so one must set the 
# CSV field size limit to the system maximum.
csv.field_size_limit(sys.maxsize)
 
# retrieves the patent assignee's organization name
def assigneeNames(patent):
    assignee_names = []
    if not len(root[patent].findall('.//assignee')):
        return 'None'
    for assignee in root[patent].findall('.//assignee'):
        name = ''.join(assignee.findtext('.//orgname'))
        assignee_names.append(name)
    return assignee_names

# 2 functions to retrieve the patent assignee's address state and country
def assigneeState(patent):
    assignee_states = []
    if not len(root[patent].findall('.//assignee')):
        return 'None'
    for assignee in root[patent].findall('.//assignee'):
        if assignee.findtext('.//state') != None:
            for state in assignee.findall('.//state'):
                state = ''.join(assignee.findtext('.//state').replace('\n', ''))
                assignee_states.append(state)
        else:
            state = 'None'
            assignee_states.append(state)
    return assignee_states

def assigneeCountry(patent):    
    assignee_countries = []
    if not len(root[patent].findall('.//assignee')):
        return 'None'
    for assignee in root[patent].findall('.//assignee'):
        if assignee.findtext('.//country') != None:
            for country in assignee.findall('.//country'):
                country = ''.join(assignee.findtext('.//country').replace('\n', ''))
                assignee_countries.append(country)
        else:
            country = 'None'
            assignee_countries.append(country)
    return assignee_countries
        
# joins all of a patent's claim text into a single string
def claimCollect(patent):
    return_claims = ''
    for claim in root[patent].findall('.//claims'):
        return_claims += ''.join(claim.itertext()).replace('\n', ' ') + ' '
    return return_claims

# Extract all relevant patent data into one list
def csvExtract(patent):
    table_row = [root[patent].findtext('.//document-id/doc-number') , \
                 root[patent].findtext('.//invention-title'), \
                 root[patent].findtext('.//document-id/date'), \
                 root[patent].findtext('.//application-reference//date'), \
                 root[patent].findtext('.//classification-national/main-classification'),\
                 len(root[patent].findall('.//references-cited/citation')), \
                 assigneeNames(patent), \
                 assigneeState(patent), \
                 assigneeCountry(patent), \
                 len(root[patent].findall('.//applicants/applicant')), \
                 claimCollect(patent)]
    return table_row 

# Extract desired data from formatted xmls and write to individual csvs
# Warning: this takes a while
# Could it be sped up with multiprocessing?
for xml in csv_source_list:
    root = ET.parse(xml).getroot()
    patent_list = root.findall('us-patent-grant')
    patents_table = []
    for index, patent in enumerate(patent_list):
        try:
            patents_table.append(csvExtract(index))
        except:
            continue
    with open(csv_destination_path + xml[-13:-4] +'.csv', 'wb') as g:
        writer = csv.writer(g)
        writer.writerows(patents_table)
    root.clear()  

# Combine all csvs into one csv
with open("../data/patents.csv", 'wb') as g:
    writer = csv.writer(g, delimiter = ';')
    for csv_file in os.listdir(csv_destination_path)[1:]:
        print 'Processing', csv_file
        with open(csv_destination_path + csv_file, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:  
                writer.writerow(row)
                
# Data Cleaning
#============================================================
headers = ['grant_doc_num', # assigned patent number
'invention_title', 
'grant_date', # when patent was granted
'appl_date', # date of patent application
'main_class', # main US government classification of patent
'num_refs', # number of references cited by the patent applicant and examiners
'assignee_org', # holder of patent
'assignee_state',
'assignee_country',
'num_applicants', # number of applicants
'claims'] # explanation of patent   

date_cols = ['grant_date', 'appl_date']
patents = pd.read_csv('../data/patents.csv', 
                      sep = ';', header = None, names = headers, 
                      parse_dates = date_cols, encoding = 'utf_8')

# Function to remove excess unicode characters from text fields
def textTrans(value):
    # Unicode value dictionary for extraneous characters 
    extraneous_table = dict.fromkeys(map(ord, '[].,"\'();'), None)    
    
    value = value.translate(extraneous_table).encode('ascii', 'ignore').lower()
    return value

# This is specifically designed to take the strings in the assignee_org column
# and remove extraneous words. I may need to modify it to work
# with the other text fields in the dataframe.
# It doesn't work 100%, but it's close enough for now.
def cleanText(value):
    company_end_list = [u'co', u'inc', u'gmbh', u'sa', u'bv', u'ltd', \
                    u'corporation', u'corp', u'limited', u'company', u'lp', \
                    u'spa', u'mfg', u'llc', u'llp', u'as', u'ag', \
                    u'aktiengesellschaft', u'sas', u'kg', u'nv', u'ab', u'sl', \
                    u'asa', u'srl', u'ltda', u'plc', u'a/s', u'jv', u'ohg', \
                    u'pty', u'incorporated', u'aktiengesekkschaft', \
                    u'aktiengesellchaft', u'aktiengesellscahft', \
                    u'aktiengesellshaft', u'aktiengessellschaft', \
                    u'aktiengsellschaft', u'ii', u'srl', u'the', u'ulc']
    converted_end_list = [word.encode('ascii', 'ignore') for word in \
    company_end_list]

    if value == 'None':
        value = np.NaN
    else:
        if len(value.split()) > 1:
            elem_list = [x for x in value.split(',')]
            name_list = []
            for name in elem_list:
                name = textTrans(name)
                split_name = name.split()
                for end in converted_end_list:
                    if end in split_name:
                        split_name.remove(end)
                        name = ' '.join(split_name)
                name_list.append(name)
            if len(name_list) > 1 and name_list[-1] is not '':
                return name_list
            else:
                return name_list[0]
        else:
            value = textTrans(value)
            if ord(value[0]) == ord(' '):
                return value.lstrip()
            else:
                return value

# Create a new column with cleaned assignee_org names  
patents['clean_company'] = patents.assignee_org.apply(lambda x: cleanText(x))

# Functions to identify patent main classification's first three characters and
# extract the classification category title from the USPTO website
r = requests.get\
('http://www.uspto.gov/web/patents/classification/selectnumwithtitle.htm')
patent_soup = BeautifulSoup(r.text, 'html.parser')

# Map the main classification numbers to the heading titles
class_num = re.compile('^[\dDGP][\dL][\dBT]')
title_list = []
for row in patent_soup.find_all('td', text = class_num):
    class_num_txt = row.text
    class_title = row.find_next('td').text
    class_dict = {class_num_txt: class_title}
    title_list.append(class_dict)

# Strip the first three characters from the main classification 
# Match the stripped character with the classication heading titles
# gathered above (needs work)
def mainClassNumber(value):
    trunc_value_list = list(value[0:3])
    if trunc_value_list[0] == ' ':
        trunc_value_list.remove(' ')
        trunc_value_list.insert(0, '0')
    elif trunc_value_list[1] == ' ':
        trunc_value_list.remove(' ')
        trunc_value_list.insert(1, '0')        
    trunc_value = ''.join(trunc_value_list)
    return trunc_value

# Gets the heading of a particular patent's classification    
def mainClassTitle(index):
    row_value = mainClassNumber(index)
    for title in title_list:
        if title.keys()[0] == row_value:
            return title.values()[0]
        else:
            pass

# Create new columns with main classification numbers and heading titles
patents['main_class_num'] = patents.main_class\
.apply(lambda x: mainClassNumber(x))   
patents['main_class_title'] = patents.main_class\
.apply(lambda x: mainClassTitle(x))

patents.dropna(subset = ['clean_company'], inplace = True)

# Write new csv with only non-null 'clean_company' values
patents.to_csv('../data/patents_notnull.csv', sep = ';', 
               encoding = 'utf_8', index = False)

# Utilize 'clean_company' names with pycrunchbase to gather funding round
# data
#============================================================
# Read in csv with non-null 'clean_company' values
date_cols = ['grant_date', 'appl_date']
patents = pd.read_csv('../data/patents_notnull.csv', sep = ';', 
                      parse_dates = date_cols, encoding = 'utf_8')
                   
# Create list of company names
indiv_comps = []
for x in patents.clean_company:
    if type(x) is not list:
        indiv_comps.append(x)

# Transform list of company names to pandas series and remove duplicates        
comp_series = pd.Series(indiv_comps)
comp_series.drop_duplicates(inplace = True)
reset = comp_series.reset_index(drop = True)

# Instantiate pycrunchbase with api key               
cb = pycrunchbase.CrunchBase(crunchbase_api_key)

# Function to gather data on funding rounds using Crunchbase uuid
def singleRound(name, uuid):
    x = cb.funding_round(uuid)
    date = x.announced_on
    if x.money_raised_usd != None:
        amount = x.money_raised_usd
    else:
        amount = np.nan
    fund_type = x.funding_type
    if x.series != None:
        series = x.series.lower()
    else: 
        series = np.nan
        
    return {'company': name, 'date': date, 'amount': amount,
            'fund_type': fund_type, 'series': series, 'uuid': uuid}

# Function that searches for companies with funding rounds listed in Crunchbase
# utilizing the singleRound function            
def cbComp(company):
    rounds = []    
    null_round = {'company': company, 'date': np.nan, 'amount': np.nan,
                  'fund_type': np.nan, 'series': np.nan, 'uuid': np.nan}
    try:
        x = cb.organization(urllib.quote(company))
    except:
        return null_round
    else:
        if bool(x.funding_rounds) != True:
            return null_round
        else:
            for event in x.funding_rounds:
                rounds.append(singleRound(company, event.uuid))
            return rounds

# Function to process company series chunks api calls to prevent timeouts and 
# data loss. Function converts each chunk to a dataframe and writes the results
# to csv.    
def downChunk(series, start, end):
    funding_rounds = []
    for company in iter(series.iloc[start:end]):
        try:
            x = cbComp(company)
        except:
            pass
        else:
            if type(x) == list:
                for funding_event in x:
                    funding_rounds.append(funding_event)
            else:
                funding_rounds.append(x)
        time.sleep(.75)
    
    rounds = pd.DataFrame(funding_rounds)

    ro_title = '../data/funding_rounds_%d_%d.csv' % (start, end - 1)
    
    rounds.to_csv(ro_title, delimiter = ';', encoding = 'utf-8', 
                  index = False)
    
    print 'Finished processing through index %d' % (end - 1)
    
# 10K unit chunks of the company series 'reset'        
chunks = [[0, 10001], [10001, 20001], [20001, 30001], [30001, 40001],
          [40001, 50001], [50001, 60001], [60001, 70001], [70001, 80001],
 [80001, len(reset)]]
 
for chunk in chunks:
    downChunk(reset, chunk[0], chunk[1])
    
# Function to combine the funding round csv chunks into a single csv
filename = '../data/funding_rounds.csv'
path = '../data/funding_rounds/'

def csvMerge(filename, path):
    with open(filename, 'wb') as g:
        writer = csv.writer(g, delimiter = ';')
        for source in os.listdir(path):
            print 'Processing ' + source
            with open(path + source, 'rU') as f:
                reader = csv.reader(f)
                reader.next()
                for row in reader:
                    writer.writerow(row)

csvMerge(filename, path)

# Separate out startup patents, startup funding rounds, series A patents
# and series A funding rounds, generate claim uniqueness value and transform
# some text variables
#===============================================================
# Read in not-null patents csv
date_cols = ['grant_date', 'appl_date']

patents = pd.read_csv('../data/patents_notnull.csv', sep = ';', 
                      parse_dates = date_cols, encoding = 'utf_8')
                      
# Read in funding rounds csv
rounds = pd.read_csv('../data/funding_rounds.csv', 
                        delimiter = ';', encoding = 'utf_8', 
                        parse_dates = 'date')

# Drop null clean_company rows from patents DF and rename column
patents.dropna(subset = ['clean_company'], inplace = True)  
patents.rename(columns = {'clean_company': 'company'}, inplace = True)

# Drop null amount and amount = 0 rows, drop duplicate uuids, and reindex
# funding rounds df
rounds.dropna(subset = ['amount'], inplace = True)
rounds.drop_duplicates(subset = ['uuid'], inplace = True)
rounds = rounds[rounds['amount'] > 0]
rounds.reset_index(drop = True, inplace = True)

# Create startup funding rounds df by extracting funding round types
# angel, seed, and venture from funding rounds df
startups = pd.concat([rounds.loc[rounds.fund_type == 'venture'], 
                      rounds.loc[rounds.fund_type == 'seed'],
                      rounds.loc[rounds.fund_type == 'angel']], axis = 0)
startups.reset_index(drop = True, inplace = True)

# Create series A df by extracting series type A from funding rounds df
series_A = rounds.loc[(rounds.series == 'a')]
series_A.reset_index(drop = True, inplace = True)

# Use set intersections to determine companies present in the patents and 
# startups dfs and the patents and series a dfs
pat_names = set(patents.company)
start_names = set(startups.company)
series_A_names = set(series_A.company)
first_names = list(pat_names & start_names)
a_names = list(pat_names & series_A_names)

# Create startup patents and series a patent df by filtering with set
# intersections
start_pats = patents.loc[patents.company.isin(first_names)]
start_pats.reset_index(drop = True, inplace = True)
a_pats = patents.loc[patents.company.isin(a_names)]
a_pats.reset_index(drop = True, inplace = True)

# Use td-idf to generate patent claim uniqueness values
# Startup patents corpus
start_corpus = []
for claim in start_pats.claims:    
    start_corpus.append(claim)

tfidf = TfidfVectorizer(strip_accents = 'unicode', stop_words = 'english')
start_term_array = tfidf.fit_transform(start_corpus).toarray()

# Uniqueness value = Sum(td-idf token values) * Number of tokens
start_tf_vals = []
for row in start_term_array:
    tf_val = np.sum(row) * np.count_nonzero(row)
    start_tf_vals.append(tf_val)

# Create new column of uniqueness values
start_pats['claim_val'] = pd.Series(start_tf_vals)

# Repeat above process for series A patents df
a_corpus = []  
for claim in a_pats.claims:    
    a_corpus.append(claim)

tfidf = TfidfVectorizer(strip_accents = 'unicode', stop_words = 'english')
a_term_array = tfidf.fit_transform(a_corpus).toarray()

a_tf_vals = []
for row in a_term_array:
    tf_val = np.sum(row) * np.count_nonzero(row)
    a_tf_vals.append(tf_val)

a_pats['claim_val'] = pd.Series(a_tf_vals)

# Deterimine the earliest patent for each company in the startup patents df
def firstPat(company):
    dates = start_pats.loc[start_pats.company == company].appl_date
    earliest = dates.min()
    return company, earliest

first_pats = []    
for company in first_names:
    first_pats.append(firstPat(company))

# Set binary column for first patents to zeros
start_pats['first_patent'] = np.zeros(len(start_pats))

# Set value of first startup patents to one; if more than one patent shares 
# earliest date, select the patent with the highest uniqueness value
for row in first_pats:
    if len(start_pats.loc[(start_pats.company == row[0]) & \
    (start_pats.appl_date == row[1])]) == 1:
        start_pats.loc[(start_pats.company == row[0]) & \
        (start_pats.appl_date == row[1]), 'first_patent'] = 1 
    else:
        most_unique = start_pats.loc[(start_pats.company == row[0]) & \
        (start_pats.appl_date == row[1])].claim_val.max()
        
        start_pats.loc[(start_pats.company == row[0]) & \
        (start_pats.appl_date == row[1]) & \
        (start_pats.claim_val == most_unique), 'first_patent'] = 1

# Filter first startup patents into separate df
first_patents = pd.DataFrame(start_pats.loc[start_pats.first_patent == 1])
first_patents.reset_index(drop = True, inplace = True)

# Hand-coded dict mapping patent main classification numbers to industry
# sector market capitalization values
sector_map = {'D10': ['Scientific & Technical Instruments ', 358], 'D16': ['Photographic Equipment & Supplies', 5],
 '036': ['Medical Appliances & Equipment', 1623], '073': ['Scientific & Technical Instruments ', 358], 
 '138': ['Industrial Equipment & Components', 41], '241': ['Farm & Construction Machinery', 1194], 
 '604': ['Medical Laboratories & Research', 224], '623': ['Medical Appliances & Equipment', 1623],
 '071': ['Agricultural Chemicals', 1189], '204': ['Specialty Chemicals', 3989],
 '424': ['Biotechnology', 11688], '435': ['Biotechnology', 11688],
 '438': ['Semiconductor Equipment & Materials', 570], '514': ['Biotechnology', 11688],
 '084': ['Internet Software & Services', 65], '310': ['Industrial Electrical Equipment', 77],
 '315': ['Electronic Equipment', 15086], '323': ['Industrial Electrical Equipment', 77],
 '330': ['Electronic Equipment', 15086], '348': ['Electronic Equipment', 15086],
 '368': ['Scientific & Technical Instruments ', 358], '370': ['Communication Equipment', 5449],
 '455': ['Communication Equipment', 5449], '600': ['Medical Laboratories & Research', 224],
 '705': ['Information Technology Services', 25294], '706': ['Information Technology Services', 25294],
 '707': ['Business Software & Services', 5692], '709': ['Application Software', 561],
 '712': ['Application Software', 561], 'D12': ['Auto Parts', 3486],
 '100': ['Farm & Construction Machinery', 1194], '192': ['Industrial Equipment & Components', 41],
 '301': ['Auto Parts', 3486], '353': ['Electronic Equipment', 15086],
 '602': ['Medical Appliances & Equipment', 1623], '117': ['Specialty Chemicals', 3989],
 '118': ['Metal Fabrication', 130], '436': ['Medical Instruments & Supplies', 288],
 '340': ['Communication Equipment', 5449], '359': ['Electronic Equipment', 15086],
 '379': ['Communication Equipment', 5449], '380': ['Security Software & Services', 36],
 '700': ['Application Software', 561], '702': ['Application Software', 561],
 '711': ['Computer Based Systems', 295], '715': ['Computer Based Systems', 295],
 'D03': ['Personal Products', 21767], '015': ['Cleaning Products', 47],
 '119': ['Housewares & Accessories', 36], '180': ['Auto Parts', 3486],
 '194': ['Processing Systems & Products', 11], '396': ['Photographic Equipment & Supplies', 5],
 '433': ['Medical Appliances & Equipment', 1623], '606': ['Medical Appliances & Equipment', 1623],
 '210': ['Scientific & Technical Instruments ', 358], '349': ['Scientific & Technical Instruments ', 358],
 '375': ['Communication Equipment', 5449], '382': ['Information Technology Services', 25294],
 '713': ['Computer Based Systems', 295], 'D11': ['Personal Products', 21767],
 '052': ['General Building Materials', 7762], '166': ['Industrial Equipment & Components', 41],
 '362': ['Electronic Equipment', 15086], '472': ['Recreational Goods, Other', 4074],
 '524': ['Synthetics', 4], '530': ['Synthetics', 4],
 '560': ['Specialty Chemicals', 3989], '367': ['Communication Equipment', 5449],
 '398': ['Communication Equipment', 5449], '607': ['Medical Appliances & Equipment', 1623],
 'D24': ['Medical Appliances & Equipment', 1623], '235': ['Communication Equipment', 5449],
 '055': ['Industrial Equipment & Components', 41], '427': ['Industrial Equipment & Components', 41],
 '257': ['Data Storage Devices', 1046], '361': ['Data Storage Devices', 1046],
 '714': ['Information Technology Services', 25294], 'D25': ['General Building Materials', 7762],
 '062': ['Appliances', 16], '239': ['General Building Materials', 7762],
 '251': ['Oil & Gas Equipment & Services', 2853], '264': ['Industrial Equipment & Components', 41],
 '426': ['Processed & Packaged Goods', 172], '536': ['Biotechnology', 11688],
 '324': ['Scientific & Technical Instruments ', 358], '365': ['Data Storage Devices', 1046],
 '385': ['Scientific & Technical Instruments ', 358], '057': ['Textile Industrial', 1443],
 '074': ['Industrial Equipment & Components', 41], '137': ['Pollution & Treatment Controls', 137],
 '048': ['Industrial Equipment & Components', 41], '148': ['Metal Fabrication', 130],
 '429': ['Industrial Equipment & Components', 41], '136': ['Industrial Electrical Equipment', 77],
 '345': ['Computer Based Systems', 295], '381': ['Computer Based Systems', 295],
 '128': ['Medical Appliances & Equipment', 1623], '190': ['Personal Products', 21767],
 'D09': ['Packaging & Containers', 1825], 'D19': ['Office Supplies', 1],
 '404': ['Heavy Construction', 852], '439': ['Networking & Communication Devices', 187],
 '463': ['Gaming Activities ', 773], '422': ['Scientific & Technical Instruments ', 358],
 '506': ['Scientific & Technical Instruments ', 358], '250': ['Industrial Electrical Equipment', 77],
 '333': ['Networking & Communication Devices', 187], '701': ['Information Technology Services', 25294],
 '049': ['Farm & Construction Machinery', 1194], '122': ['Industrial Equipment & Components', 41],
 '358': ['Computer Based Systems', 295], '372': ['Scientific & Technical Instruments ', 358],
 '139': ['Textile Industrial', 1443], '451': ['Industrial Equipment & Components', 41],
 '540': ['Synthetics', 4], '326': ['Semiconductor - Specialized', 74],
 '710': ['Computer Based Systems', 295], '717': ['Computer Based Systems', 295],
 '269': ['Industrial Equipment & Components', 41], '095': ['Industrial Equipment & Components', 41],
 '428': ['Industrial Equipment & Components', 41], '356': ['Scientific & Technical Instruments', 358],
 '454': ['Industrial Equipment & Components', 41], '208': ['Synthetics', 4],
 '518': ['Synthetics', 4], '704': ['Computer Based Systems', 295],
 '060': ['Electric Utilities', 4499], '206': ['Packaging & Containers', 1825],
 '222': ['Housewares & Accessories', 36], '482': ['Appliances', 16],
 '044': ['Oil & Gas Refining & Marketing', 7335], '099': ['Housewares & Accessories', 36],
 '415': ['Industrial Equipment & Components', 41], '564': ['Biotechnology', 11688],
 '716': ['Computer Based Systems', 295], '248': ['Small Tools & Accessories', 19],
 '273': ['Toys & Games', 16], '423': ['Specialty Chemicals', 3989],
 '548': ['Biotechnology', 11688], '703': ['Computer Based Systems', 295],
 '708': ['Computer Based Systems', 295], '261': ['Housewares & Accessories', 36],
 '280': ['Housewares & Accessories', 36], '331': ['Scientific & Technical Instruments', 358],
 '029': ['Metal Fabrication', 130], '209': ['Industrial Equipment & Components', 41],
 '546': ['Biotechnology', 11688], '219': ['Industrial Equipment & Components', 41],
 '0 2': ['Textile - Apparel Clothing', 849], '108': ['Home Furnishings & Fixtures', 435],
 '283': ['Paper & Paper Products', 901], '343': ['Communication Equipment', 5449],
 '363': ['Processing Systems & Products', 11], '351': ['Medical Instruments & Supplies', 288],
 '374': ['Scientific & Technical Instruments', 358], '442': ['Textile - Apparel Clothing', 849],
 '525': ['Synthetics', 4], '434': ['Electronic Equipment', 15086],
 '307': ['Industrial Electrical Equipment', 77], '417': ['Industrial Equipment & Components', 41],
 '201': ['Waste Management', 259], '549': ['Biotechnology', 11688],
 '342': ['Communication Equipment', 5449], 'D13': ['Industrial Electrical Equipment', 77],
 '114': ['Industrial Equipment & Components', 41], '106': ['Synthetics', 4],
 '341': ['Communication Equipment', 5449], 'D08': ['Small Tools & Accessories', 19],
 'D14': ['Communication Equipment', 5449], '156': ['Synthetics', 4],
 '244': ['Aerospace/Defense Products & Services', 462], 'D02': ['Textile - Apparel Footwear & Accessories', 941],
 '105': ['Railroads', 919], '220': ['Housewares & Accessories', 36],
 '312': ['Home Furnishings & Fixtures', 435], '336': ['Industrial Electrical Equipment', 77],
 '135': ['Housewares & Accessories', 36], '290': ['Industrial Electrical Equipment', 77],
 '450': ['Textile - Apparel Footwear & Accessories', 941], '313': ['Housewares & Accessories', 36],
 '320': ['Electronic Equipment', 15086], '425': ['Rubber & Plastics', 1035],
 '376': ['Electric Utilities', 4499], '221': ['Pollution & Treatment Controls', 137],
 '347': ['Paper & Paper Products', 901], '601': ['Medical Appliances & Equipment', 1623],
 '165': ['Industrial Equipment & Components', 41], '508': ['Oil & Gas Refining & Marketing', 7335],
 '110': ['Industrial Equipment & Components', 41], '327': ['Computer Based Systems', 295],
 'D07': ['Food - Major Diversified', 7910], '294': ['Packaging & Containers', 1825],
 '800': ['Biotechnology', 11688], '202': ['Pollution & Treatment Controls', 137],
 '252': ['Synthetics', 4], 'D06': ['Housewares & Accessories', 36],
 '141': ['Home Furnishings & Fixtures', 435], '297': ['Home Furnishings & Fixtures', 435],
 'D21': ['Toys & Games', 16], '0 5': ['Housewares & Accessories', 36],
 '182': ['General Building Materials', 7762], '228': ['Metal Fabrication', 130],
 '502': ['Synthetics', 4], '096': ['Pollution & Treatment Controls', 137],
 '431': ['Industrial Equipment & Components', 41], '053': ['Packaging & Containers', 1825],
 '215': ['Packaging & Containers', 1825], '526': ['Synthetics', 4],
 '123': ['Auto Parts', 3486], '440': ['Auto Parts', 3486],
 '477': ['Auto Parts', 3486], '386': ['Electronic Equipment', 15086],
 'D30': ['Farm & Construction Machinery', 1194], '409': ['Machine Tools & Accessories', 82],
 '446': ['Toys & Games', 16], '544': ['Biotechnology', 11688],
 '528': ['Rubber & Plastics', 1035], '0 4': ['Housewares & Accessories', 36],
 '126': ['Industrial Equipment & Components', 41], '134': ['Cleaning Products', 47],
 '430': ['Medical Instruments & Supplies', 288], 'D34': ['Packaging & Containers', 1825],
 '475': ['Industrial Equipment & Components', 41], '174': ['Industrial Electrical Equipment', 77],
 '070': ['Home Furnishings & Fixtures', 435], '033': ['Scientific & Technical Instruments', 358],
 '493': ['Packaging & Containers', 1825], '392': ['Industrial Equipment & Components', 41],
 '089': ['Aerospace/Defense Products & Services', 462], '378': ['Scientific & Technical Instruments', 358],
 'D26': ['Recreational Goods, Other', 4074], '042': ['Aerospace/Defense Products & Services', 462],
 'D23': ['Appliances', 16], '047': ['Farm Products', 924],
 '205': ['Industrial Equipment & Components', 41], '369': ['Data Storage Devices', 1046],
 '216': ['Housewares & Accessories', 36], '177': ['Scientific & Technical Instruments', 358],
 '718': ['Computer Based Systems', 295], '414': ['Industrial Equipment & Components', 41],
 'D15': ['Housewares & Accessories', 36], '416': ['Industrial Equipment & Components', 41],
 '335': ['Electronic Equipment', 15086], '132': ['General Building Materials', 7762],
 '504': ['Agricultural Chemicals', 1189], '040': ['Paper & Paper Products', 901],
 '198': ['Industrial Equipment & Components', 41], '445': ['Auto Parts', 3486],
 '453': ['Processing Systems & Products', 11], '554': ['Biotechnology', 11688],
 '224': ['Packaging & Containers', 1825], '406': ['Industrial Equipment & Components', 41],
 '101': ['Paper & Paper Products', 901], '521': ['Rubber & Plastics', 1035],
 '030': ['Housewares & Accessories', 36], '494': ['Scientific & Technical Instruments', 358],
 '405': ['Farm & Construction Machinery', 1194], '562': ['Biotechnology', 11688],
 '267': ['Textile - Apparel Clothing', 849], '503': ['Electronic Equipment', 15086],
 '024': ['Textile - Apparel Clothing', 849], '056': ['Farm & Construction Machinery', 1194],
'366': ['Industrial Equipment & Components', 41], '373': ['Industrial Equipment & Components', 41],
'522': ['Biotechnology', 11688]}

# Create new columns in first startup patents df for industry sector name 
# and market cap
first_patents['market_sector'] = first_patents.main_class_num\
.map(lambda x: sector_map[x][0])
first_patents['sector_size'] = first_patents.main_class_num\
.map(lambda x: sector_map[x][1])

# Create binary column in startup funding rounds df for earliest round
startups['first_round'] = np.zeros(len(startups))

def firstRound(company):
    dates = startups.loc[startups.company == company].date
    earliest = dates.min()
    return company, earliest
   
early_rounds = []
for company in first_names:
    early_rounds.append(firstRound(company))

# Set value of first startup funding rounds to one; if more than one round 
# earliest date, select the patent with the highest uniqueness value    
for row in early_rounds:
    if len(startups.loc[(startups.company == row[0]) & \
    (startups.date == row[1])]) == 1:
        startups.loc[(startups.company == row[0]) & \
        (startups.date == row[1]), 'first_round'] = 1
    else:
        max_val = startups.loc[(startups.company == row[0]) & \
        (startups.date == row[1])].amount.max()
        startups.loc[(startups.company == row[0]) & \
        (startups.date == row[1]) & (startups.amount == max_val), \
        'first_round'] = 1

# Create separate df for earliest funding rounds    
first_rounds = pd.DataFrame(startups.loc[startups.first_round == 1])
first_rounds.reset_index(drop = True, inplace = True)

# Merge first patents df with first funding rounds df
first_patents = first_patents.merge(first_rounds, on = 'company')    

# Remove extraneous characters from assignee state and country columns
first_patents.assignee_state = first_patents.assignee_state\
.apply(lambda x: textTrans(x))
first_patents.assignee_country = first_patents.assignee_country\
.apply(lambda x: textTrans(x))

# Remove unnecessary columns
del first_patents['appl_date']
del first_patents['first_patent']
del first_patents['uuid']
del first_patents['first_round']
del first_patents['claims']

# Write first patents/funding rounds df to csv
first_patents.to_csv('../data/first_patents.csv', sep = ';', 
                     encoding = 'utf_8', index = False)
                     
# Repeat process of finding earliest patents and funding rounds for 
# series A patent and funding round dfs
def firstARound(company):
    a_round = pd.to_datetime(series_A[series_A.company == company].date.min())
    return a_round                     

close_rounds = []
for name in a_names:
    y = firstARound(name)
    pair = [name, str(y)]
    close_rounds.append(pair)

series_A['first_a_round'] = np.zeros(len(series_A))

for row in close_rounds:
    if len(series_A.loc[(series_A.company == row[0]) & \
    (series_A.date == row[1])]) == 1:
        series_A.loc[(series_A.company == row[0]) & \
        (series_A.date == row[1]), 'first_a_round'] = 1
    
    else:
        max_val = series_A.loc[(series_A.company == row[0]) & \
        (series_A.date == row[1])].amount.max()
        
        series_A.loc[(series_A.company == row[0]) & \
        (series_A.date == row[1]) & (series_A.amount == max_val), \
        'first_a_round'] = 1
                     
def closest(company):
    pats = patents.loc[patents.company == company].sort('appl_date').appl_date.iloc[:10]
    frnd = firstARound(company)
    
    closest = pats.iloc[0]
    diff = abs(pats.iloc[0] - frnd)    
    for patent in pats:        
        if abs(patent - frnd) < diff:
            diff = abs(patent - frnd)
            closest = patent        
    return closest
        
close_pats = []
for name in a_names:
    x = closest(name)
    pair = [name, x]
    close_pats.append(pair)   
    
a_pats['series_a_patent'] = np.zeros(len(a_pats))

for row in close_pats:
    if len(a_pats.loc[(a_pats.company == row[0]) & \
    (a_pats.appl_date == row[1])]) == 1:
        a_pats.loc[(a_pats.company == row[0]) & \
        (a_pats.appl_date == row[1]), 'series_a_patent'] = 1
    
    else:
        most_unique = a_pats.loc[(a_pats.company == row[0]) & \
        (a_pats.appl_date == row[1])].claim_val.max()
        
        a_pats.loc[(a_pats.company == row[0]) & \
        (a_pats.appl_date == row[1]) & \
        (a_pats.claim_val == most_unique), 'series_a_patent'] = 1

series_a_rounds = series_A.loc[series_A.first_a_round == 1]
series_a_patents = a_pats.loc[a_pats.series_a_patent == 1]

series_a = series_a_patents.merge(series_a_rounds, on = 'company')

del series_a['grant_doc_num']
del series_a['grant_date']
del series_a['assignee_org']
del series_a['series_a_patent']
del series_a['fund_type']
del series_a['series']
del series_a['uuid']
del series_a['first_a_round']

series_a.assignee_state = series_a.assignee_state\
.apply(lambda x: textTrans(x))
series_a.assignee_country = series_a.assignee_country\
.apply(lambda x: textTrans(x))
series_a['market_sector'] = series_a.main_class_num\
.map(lambda x: sector_map[x][0])
series_a['sector_size'] = series_a.main_class_num\
.map(lambda x: sector_map[x][1])

series_a.to_csv('../data/series_a.csv', sep = ';', encoding = 'utf_8',
                     index = False)
                     
# Analysis
#===============================================================
# Read in first patents df
first_patents = pd.read_csv('../data/first_patents.csv', sep = ';', 
                            parse_dates = 'date', encoding = 'utf_8')

# Reorder columns
first_patents = first_patents[['amount', 'company', 'invention_title', 'appl_date',
                               'main_class_title', 'main_class_num', 'fund_type',
                               'series', 'assignee_state', 'assignee_country',
                               'market_sector', 'sector_size', 'num_refs',
                               'num_applicants', 'claim_val']]

first_patents.shape

# Create visualizations of key variables
first_patents.main_class_num.value_counts().apply(np.log).hist(bins = 20, 
grid = False, color = sns.xkcd_rgb["mid blue"])
plt.title('Main Patent Classification Distribution', size = 12)
plt.xlabel('log(number of patents sharing same main classification number)')
plt.ylabel('count')
sns.despine()

first_patents.amount.apply(np.log).hist(bins = 50, grid = False, 
color = sns.xkcd_rgb["deep red"])
plt.title('Distribution of Funding Round Amounts (Response)', size = 12)
plt.xlabel('log(funding round amount in USD)')
plt.ylabel('count')
sns.despine()

first_patents.claim_val.apply(np.log).hist(bins = 20, grid = False, 
color = sns.xkcd_rgb["soft purple"])
plt.title('Distribution of Uniqueness Values', size = 12)
plt.xlabel('log(uniqueness value of patent claim text)')
plt.ylabel('count')
sns.despine()

first_patents.sector_size.apply(np.log).hist(grid = False, 
color = sns.xkcd_rgb["light orange"])
plt.title('Distribution of Market Sector Size', size = 12)
plt.xlabel('log(industry sector market capitalization)')
plt.ylabel('count')
sns.despine()

first_patents.num_applicants.hist(grid = False, 
                                  color = sns.xkcd_rgb["wine"])
plt.title('Distribution of Number of Patent Applicants', size = 12)
plt.xlabel('number of patent applicants')
plt.ylabel('count')
sns.despine()

first_patents.num_refs.hist(grid = False, color = sns.xkcd_rgb["denim"])
plt.title('Distribution of Number of Patent References', size = 12)
plt.xlabel('number of patent references')
plt.ylabel('count')
sns.despine()

# Create dummy variables for categorical features
class_dummies = pd.get_dummies(first_patents.main_class_num, 
                               prefix = 'm_dum').iloc[:, 1:]
first_patents = pd.concat([first_patents, class_dummies], axis = 1)

state_dummies = pd.get_dummies(first_patents.assignee_state, 
                               prefix = 's_dum').iloc[:, 1:]
first_patents = pd.concat([first_patents, state_dummies], axis = 1)

country_dummies = pd.get_dummies(first_patents.assignee_country, 
                                 prefix = 'c_dum').iloc[:, 1:]
first_patents = pd.concat([first_patents, country_dummies], axis = 1)

# Create list of feature columns
col_list = list(first_patents.columns.values)
feature_cols = col_list[11:]

# Random Forest Regression_________________________________________________

# Feature matrix with all first patent features
rf_X = first_patents[feature_cols]
# Feature matrix of only uniqueness value, market sector size, number of 
# applicants and number of references
rf_X2 = first_patents[feature_cols[:4]]
# Response series of logged first patent amount values
rf_y = first_patents.amount.apply(np.log)

# Random forest GridSearchCV, all features
rf = RandomForestRegressor(max_features = 'auto', random_state = 1, n_jobs = 3)
max_n_range = range(100, 850, 50)
param_grid = dict(n_estimators = max_n_range)
rf_grid = GridSearchCV(rf, param_grid, cv = 10, scoring = 'mean_squared_error')
rf_grid.fit(rf_X, rf_y)
rf_grid.best_params_ # {'n_estimators': 600}
np.sqrt(-rf_grid.best_score_) # 1.7450179962896533

# Train-test-split of all features
rf_X_train, rf_X_test, rf_y_train, rf_y_test = train_test_split(rf_X, rf_y, 
                                                                random_state = 1)
# Random forest with optimal parameter from grid search
rf1 = RandomForestRegressor(n_estimators = 600, max_features = 'auto', 
                           oob_score = True, random_state = 1, n_jobs = 3)
rf1.fit(rf_X_train, rf_y_train)
rf1.oob_score_ # -0.11477648832224174
preds = rf1.predict(rf_X_test)
np.sqrt(metrics.mean_squared_error(rf_y_test, preds)) # RMSE = 1.7160247741158297
rf1_features = pd.DataFrame({'feature':feature_cols, 
'importance':rf1.feature_importances_})
rf1_features.sort('importance', ascending = False).head(10)

# Random forest GridSearchCV, four features
rf = RandomForestRegressor(max_features = 'auto', random_state = 1, 
                           n_jobs = 3)
max_n_range = range(100, 850, 50)
max_leaf_node_range = range(2, 11)
param_grid = dict(n_estimators = max_n_range, max_leaf_nodes = max_leaf_node_range)
rf_grid = GridSearchCV(rf, param_grid, cv = 10, scoring = 'mean_squared_error')
rf_grid.fit(rf_X2, rf_y)
rf_grid.best_params_ # {'max_leaf_nodes': 6, 'n_estimators': 800}
np.sqrt(-rf_grid.best_score_) # 1.6672775471011514

# Train-test-split of four features
rf_X2_train, rf_X2_test, rf_y_train, rf_y_test = train_test_split(rf_X2, rf_y,
                                                                    random_state = 1)
# Random forest with optimal parameter from grid search
rf2 = RandomForestRegressor(n_estimators = 800, max_features = 'auto', 
                            max_leaf_nodes = 6, oob_score = True, 
                            random_state = 1, n_jobs = 3)
rf2.fit(rf_X2_train, rf_y_train)
rf2.oob_score_ # 0.0021809859111234786
preds = rf2.predict(rf_X2_test)
np.sqrt(metrics.mean_squared_error(rf_y_test, preds)) # RMSE = 1.6303398440226933
rf2_features = pd.DataFrame({'feature':feature_cols[:4], 
'importance':rf2.feature_importances_})
rf2_features.sort('importance', ascending = False).head(10)

# Focus on Series A funding rounds only

# Read in series a patent/funding rounds df
a_patents = pd.read_csv('../data/series_a.csv', sep = ';', encoding = 'utf_8')

del a_patents['main_class']
del a_patents['claims']
del a_patents['date']

# Reorder columns
a_patents = a_patents[['amount', 'company', 'invention_title', 'appl_date', 
'main_class_title', 'main_class_num', 'assignee_state', 'assignee_country', 
'market_sector', 'sector_size', 'num_refs','num_applicants', 'claim_val']]

# Visualize key variables
a_patents.main_class_num.value_counts().apply(np.log).hist(bins = 15, 
grid = False, color = sns.xkcd_rgb["blue"])
plt.xlabel('log(number of patents sharing same main classification number)')
plt.ylabel('count')
sns.despine()

a_patents.amount.apply(np.log).hist(bins = 40, grid = False, 
color = sns.xkcd_rgb["deep red"])
plt.xlabel('log(funding round amount in USD)')
plt.ylabel('count')
sns.despine()

a_patents.claim_val.hist(bins = 20, grid = False, color = sns.xkcd_rgb["slate"])
plt.xlabel('tfidf value of patent claim text')
plt.ylabel('count')
sns.despine()

a_patents.num_refs.hist(grid = False, color = sns.xkcd_rgb["greenish"])
plt.xlabel('number of references cited in patent')
plt.ylabel('count')
sns.despine()

a_patents.num_applicants.hist(grid = False, color = sns.xkcd_rgb["wine"])
plt.xlabel('number of patent applicants')
plt.ylabel('count')
sns.despine()

# Create dummies of categorical features                            
a_class_dummies = pd.get_dummies(a_patents.main_class_num, 
                                 prefix = 'm_dum').iloc[:, 1:]
a_patents = pd.concat([a_patents, a_class_dummies], axis = 1)

a_state_dummies = pd.get_dummies(a_patents.assignee_state, 
                                 prefix = 's_dum').iloc[:, 1:]
a_patents = pd.concat([a_patents, a_state_dummies], axis = 1)

a_country_dummies = pd.get_dummies(a_patents.assignee_country, 
                                   prefix = 'c_dum').iloc[:, 1:]
a_patents = pd.concat([a_patents, a_country_dummies], axis = 1)

# Create feature list
a_col_list = list(a_patents.columns.values)
a_feature_cols = a_col_list[9:]

# Random Forest Regression_________________________________________________
# Feature matrix with all series a features
arf_X = a_patents[a_feature_cols]
# Feature matrix of only uniqueness value, market sector size, number of 
# applicants and number of references
arf_X2 = a_patents[a_feature_cols[:4]]
# Response series of logged series a amount values
arf_y = a_patents.amount.apply(np.log)

# Random forest GridSearchCV, all features
arf = RandomForestRegressor(max_features = 'auto', random_state = 1, 
                            n_jobs = 3)
max_n_range = range(100, 850, 50)
param_grid = dict(n_estimators = max_n_range)
arf_grid = GridSearchCV(arf, param_grid, cv = 10, scoring = 'mean_squared_error')
arf_grid.fit(arf_X, arf_y)
arf_grid.best_params_ # {'n_estimators': 700}
np.sqrt(-arf_grid.best_score_) # 1.0426139003156252

# Train-test-split of all features
arf_X_train, arf_X_test, arf_y_train, arf_y_test = train_test_split(arf_X, arf_y,
                                                                    random_state = 1)
# Random forest with optimal parameter from grid search
arf_1 = RandomForestRegressor(n_estimators = 700, max_features = 'auto', 
                           oob_score = True, random_state = 1, n_jobs = 3)
arf_1.fit(arf_X_train, arf_y_train)
arf_1.oob_score_ # -0.12759478764657217
preds = arf_1.predict(arf_X_test)
np.sqrt(metrics.mean_squared_error(arf_y_test, preds)) # RMSE = 1.0070616514513329
arf_1_features = pd.DataFrame({'feature':a_feature_cols, 
'importance':arf_1.feature_importances_})
arf_1_features.sort('importance', ascending = False).head(10)

# Random forest GridSearchCV, four features
arf2 = RandomForestRegressor(max_features = 'auto', random_state = 1, 
                             n_jobs = 3)
max_n_range = range(100, 850, 50)
max_leaf_node_range = range(2, 11)
param_grid = dict(n_estimators = max_n_range, max_leaf_nodes = max_leaf_node_range)
arf2_grid = GridSearchCV(arf2, param_grid, cv = 10, scoring = 'mean_squared_error')
arf2_grid.fit(arf_X2, arf_y)
arf2_grid.best_params_ # {'max_leaf_nodes': 2, 'n_estimators': 100}
np.sqrt(-arf2_grid.best_score_) # 0.97704105408210207

# Train-test-split of four features
arf_X2_train, arf_X2_test, arf_y_train, arf_y_test = train_test_split(arf_X2, arf_y,
                                                                    random_state = 1)
# Random forest with optimal parameter from grid search
arf_2 = RandomForestRegressor(n_estimators = 100, max_features = 'auto', 
                              max_leaf_nodes = 2, oob_score = True, 
                              random_state = 1, n_jobs = 3)
arf_2.fit(arf_X2_train, arf_y_train)
arf_2.oob_score_ # -0.002899632678329489
preds = arf_2.predict(arf_X2_test)
np.sqrt(metrics.mean_squared_error(arf_y_test, preds)) # RMSE = 0.9626713483785625
arf_2_features = pd.DataFrame({'feature':a_feature_cols[:4], 
'importance':arf_2.feature_importances_})
arf_2_features.sort('importance', ascending = False).head(10)
