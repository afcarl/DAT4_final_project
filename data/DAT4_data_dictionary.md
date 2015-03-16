## Extracted Patent Data Dictionary (patents\_notnull\_sample.csv)
* Official documentation available in the [Patent Green Book](http://storage.googleapis.com/patents/docs/PatentFullTextAPSDoc_GreenBook.pdf)
* grant\_doc\_num : assigned patent number
* invention\_title : invention title
* grant\_date : date of patent grant
* appl\_date : date of patent application
* main\_class :  [main USPTO classification](http://www.uspto.gov/web/patents/classification/selectnumwithtitle.htm) of patent
* num\_refs :  number of references cited by the patent applicant and examiners
* assignee_org :  holder of patent
* assignee_state :  given state location of patent holder
* assignee_country :  given country location of patent holder
* num_applicants :  number of applicants
* claims :  explanation of patent (full text)

## Combined Patent and Funding Round Data Dictionary (first\_patents.csv & series\_a.csv)
* amount :  funding round amount in USD
* company :  name of company holding patent and receiving funding
* invention_title :  invention title
* appl_date :  date of patent application
* main\_class\_title :  [main USPTO classification heading title](http://www.uspto.gov/web/patents/classification/selectnumwithtitle.htm) of patent main classification
* main\_class\_num :  [main USPTO classification heading number](http://www.uspto.gov/web/patents/classification/selectnumwithtitle.htm) of patent main classification
* fund_type :  categorization of funding round: angel, seed, venture
* series :  series a-h, where applicable and available
* market_sector :  handcoded most applicable industry sector based on patent main classification, source [Yahoo! Finance](https://biz.yahoo.com/ic/ind_index.html)
* sector_size :  size of industry sector in billions USD
* num_refs :  number of references cited by the patent applicant and examiners
* num_applicants :  number of applicants
* claim_val :  patent claim uniqueness value (sum(tf-idf values) * (# non-stop-word tokens))
