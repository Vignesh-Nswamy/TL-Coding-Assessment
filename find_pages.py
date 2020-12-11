import re
import io
import sys
import gzip
import pickle
import requests
import traceback
import langdetect
from tqdm import tqdm
from langdetect import detect
from bs4 import BeautifulSoup
from textblob import TextBlob
from warcio.archiveiterator import ArchiveIterator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

title_re = re.compile("<title>(.+?)</title>")

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
                    if title_re.search(html_content) is not None:
                        title = title_re.search(html_content).group(1)
                        try:
                            if detect(title) == 'en' and ('covid' in title.lower() or 'corona' in title.lower() or 'pandemic' in title.lower()):
                                yield title, page_uri
                        except langdetect.lang_detect_exception.LangDetectException:
                            # traceback.print_exc()
                            continue


search_str = 'Economic impact of Covid-19'
relevent_uri = list()
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

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
                for title, page_uri in tqdm(get_titles_from_warc(warc_uri)):
                    sentences = [
                        title,
                        search_str
                    ]
                    sen_embeddings = model.encode(sentences)
                    cos_sim = cosine_similarity(sen_embeddings)
                    if cos_sim[0][1] > 0.75:
                        print(page_uri)
                        relevent_uri.append(page_uri)
                
                if len(relevent_uri) > 100:
                    break
    except:
        with open('relevant_uri.pkl', 'wb') as f:
            pickle.dump(relevent_uri, f)
            traceback.print_exc()
        sys.exit(0)

with open('relevant_uri.pkl', 'wb') as f:
    pickle.dump(relevent_uri, f)