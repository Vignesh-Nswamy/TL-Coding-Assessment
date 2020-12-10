import re
import io
import sys
import gzip
import string
import pickle
import requests
import traceback
import langdetect
from tqdm import tqdm
from langdetect import detect
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from warcio.archiveiterator import ArchiveIterator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 

import nltk
nltk.download('stopwords')

porter = PorterStemmer()
regex = re.compile('[%s]' % re.escape(string.punctuation))
sw = stopwords.words("english")

def clean_text(text):
    text = text.lower()
    text = regex.sub('', text)
    text = ' '.join([word for word in text.split() if (len(word)>=4)])
    text = ' '.join([word for word in text.split() if word not in sw])
    text = ' '.join([porter.stem(word) for word in text.split()])
    return text

def get_page_content(html_page):
    if html_page.startswith('https') or html_page.startswith('http'):
        html_page = requests.get(html_page).content
    soup = BeautifulSoup(html_page, 'html.parser')
    return clean_text(soup.text)

ref_docs = [
    get_page_content('https://en.wikipedia.org/wiki/Economic_impact_of_the_COVID-19_pandemic'),
    get_page_content('https://www.who.int/news/item/13-10-2020-impact-of-covid-19-on-people\'s-livelihoods-their-health-and-our-food-systems'),
    get_page_content('https://www.brookings.edu/research/ten-facts-about-covid-19-and-the-u-s-economy/'),
    get_page_content('https://www.mckinsey.com/business-functions/risk/our-insights/covid-19-implications-for-business'),
    get_page_content('https://carsey.unh.edu/COVID-19-Economic-Impact-By-State'),
    get_page_content('https://www.frontiersin.org/articles/10.3389/fpubh.2020.00241/full'),
    get_page_content('https://www.pewsocialtrends.org/2020/09/24/economic-fallout-from-covid-19-continues-to-hit-lower-income-americans-the-hardest/'),
    get_page_content('https://www.reuters.com/article/us-usa-economy-poll/u-s-economy-to-slow-in-first-quarter-but-reach-pre-covid-19-levels-in-a-year-reuters-poll-idUSKBN28K00A'),
    get_page_content('https://www.mckinsey.com/business-functions/strategy-and-corporate-finance/our-insights/the-coronavirus-effect-on-global-economic-sentiment')
]

vectorizer = TfidfVectorizer()
docs_tfidf = vectorizer.fit_transform(ref_docs)


url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def get_titles_from_warc(url):
    resp = requests.get(url, stream=True)

    for record in ArchiveIterator(resp.raw, arc2warc=True):
        if record.rec_type == 'warcinfo':
            continue
        
        if re.match(url_regex, record.rec_headers.get_header("WARC-Target-URI")) is None:
            continue

        elif record.rec_type is not None and record.rec_type == 'response':
            if record.http_headers is not None and  record.http_headers.get_header('Content-Type') is not None and record.http_headers.get_header('Content-Type') == 'text/html':
                html_content = record.content_stream().read().decode("utf-8", "replace")
                if html_content is not None:
                    page_uri = record.rec_headers.get_header('WARC-Target-URI')
                    clean_text = get_page_content(html_content)
                    return vectorizer.transform([clean_text]), page_uri


search_str = 'Economic impact of Covid-19'
relevent_uri = list()

warc_list = [
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/warc.paths.gz',
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-45/warc.paths.gz',
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-40/warc.paths.gz',
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-34/warc.paths.gz',
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-29/warc.paths.gz',
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-24/warc.paths.gz',
    'https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-16/warc.paths.gz'
]

for warc_all_uri in warc_list:
    try:
        web_response = requests.get(warc_all_uri, stream=True)
        gz_file = web_response.content

        f = io.BytesIO(gz_file)
        with gzip.GzipFile(fileobj=f) as fh:
            for incomplete_uri in fh:
                incomplete_uri = incomplete_uri.decode().replace('\n', '')
                warc_uri = f'https://commoncrawl.s3.amazonaws.com/{incomplete_uri}'
                print(f'Extracting from: {warc_uri}')
                for doc_vector, page_uri in tqdm(get_titles_from_warc(warc_uri)):
                    cos_sim = np.average(cosine_similarity(doc_vector, docs_tfidf).flatten())
                    if cos_sim[0][1] > 0.3:
                        print(page_uri)
                        relevent_uri.append(page_uri)
                
                if len(relevent_uri) > 1000:
                    break
    except:
        with open('relevant_uri.pkl', 'wb') as f:
            pickle.dump(relevent_uri, f)
            traceback.print_exc()
        sys.exit(0)

with open('relevant_uri.pkl', 'wb') as f:
    pickle.dump(relevent_uri, f)