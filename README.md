# Web URI extraction based on relevance to query - TLinternship2020

This is a summary of the work that has been done in this project to fetch URIs whose 
contents match a given query string.
Multiple approaches, explained in detail below,
have been explored to solve the given problem. 

## Dataset

The data used in this project contains information about billions of web pages in WARC format
available for anyone to use at https://commoncrawl.org/

### Data Cleaning

The data extracted from WARC files have been filtered and cleaned with extreme care so as to maintain its 
integrity and usability.
**Invalid URIs are filtered out using a regular expression** to prevent the code from sifting through hundreds or
perhaps thousands of junk URI. Apart from that, only those records of type 'response'
and with english content have been taken for further processing.

## Approaches

### Identifying matching URIs using the cosine similarities between web pages' title and the given query string

The core idea of this approach is that, **in an embedding space, similar objects 
lie close to eac other**. Extending this to web pages, we can say that,
web **pages with similar content, and this similar titles, when embedded into an 
n-dimensional vector, are not far away from each other**.

For the sake of simplicity and for direct comparison with embeddings of the query string,
in this approach, we encode only the title of each web page using 
a sentence BERT with a bert-large backbone capable of capturing the meaning
of an entire sentence in a single vector.

In the notebook ```demo_title_vs_query.ipynb```, the viability of this approach
has been explored. A set of related and unrelated web pages were gathered,
their titles extracted using a regular expression parser and embedded and a similarity matrix was computed.
The matrix clearly seemed to support the hypothesis with encouraging results.

The files ```find_pages.py``` and ```find_pages.ipynb``` use the above technique 
to find and store web URIs with relevant content to query string.

The titles and query strings are encoded using ```encode()``` function 
of ```SentenceTransformer``` and pairwise cosine similarities found using ```sklearn```'s
```cosine_similarity``` function

The results would depend on how high or low the ```similarity threshold``` 
is set. Set it too low, result would have irrelevant articles. Set it too high,
result will not be substantial.

The **WARC files** used here are of the months **November/December 2020 and October 2020**.

### Utilizing cosine similarities between embeddings of entire web page content to find best matches with query string

This approach takes the concepts of the previous approach and **extends it to 
and entire document or article instead of just the title** of a web page.

However, to keep things interesting, ```tf-idf document vectors``` are used instead of 
sentence BERT encodings. This approach bypasses one of the very few disadvantages of BERT, that is,
limitation of number of tokens that can be fed into the model.

One more major change in this approach is that, instead of using the 
query string directly, a corpus is created with several gold standard reference 
articles that the web pages can be compared to. 

The web URIs pass through the same rigorous filtering process as before. But instead of 
extracting the title, html content of the web page is parsed by ```BeautifulSoup```.
From that, textual content is extracted to go through further cleaning to remove stop words,
stem and lemmatize each word. This is done to build a more robust tf-idf vectorizer.
After that, the content is embedded using the tf-idf vectorizer and its similarity with
all the gold standard articles is computed and averaged. Depending on this similarity, 
they are either discarded or added to relevant articles.

Ideally, this approach should've been better than the former because web page
contents are embedded completely thus having embeddings that capture more meaning and context.
However, in practice, due to the data format, it is extremely expensive to clean and extract text
from the html content. This proved to be a major disadvantage of this approach.

### Other worthy mentions

#### Embedding content summary instead of an entire web page

The summary of each web pages' content can be used along with a sentence BERT instead of 
tf-idf vectorizers. This approach might suffer from the same overhead as extracting embeddings
of entire web page content.

#### Embedding all web page titles at once and using kNN to find the most similar web pages

This approach would be an inexpensive experiment and definitely worth a shot.

## Conclusion

Comparing the two major approaches, **Identifying matching URIs using the cosine similarities between web pages' title and the given query string**
 seems to be the clear winner because it is not bottlenecked by an expensive text
extraction and cleaning process.




