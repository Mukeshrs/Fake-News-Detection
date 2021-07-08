import requests
import bs4

# Make two strings with default google search URL
# 'https://google.com/search?q=' and
# our customized search keyword.
# Concatenate them
text = "India plans on going to war with USA "
url = 'https://google.com/search?q=news' + text

# Fetch the URL data using requests.get(url),
# store it in a variable, request_result.
request_result = requests.get(url)

# Creating soup from the fetched request
soup = bs4.BeautifulSoup(request_result.text,
                         "html.parser")
#print(soup)

# soup.find.all( h3 ) to grab
# all major headings of our search result,
heading_object = soup.find_all('h3')

# Iterate through the object
# and print it as a string.
lst= []
lst.append(text)

for info in heading_object:
    #print(info.getText())
    lst.append(info.getText())

#print(lst)

import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

cleaned = list(map(clean_string, lst))
print(cleaned)

vectorizer = CountVectorizer().fit_transform(cleaned)
vectors = vectorizer.toarray()
#print(vectors)



def cosine_sim_vectors(vec1,vec2):
    vec1 = vec1.reshape(1,-1)
    vec2 = vec2.reshape(1,-1)
    return cosine_similarity(vec1,vec2)

print(cosine_sim_vectors(vectors[0],vectors[1]))

cmp = vectors[0]
sim_lst=[]

for i in range(1, len(vectors)):
    cal=cosine_sim_vectors(cmp,vectors[i])
    sim_lst.append(cal[0][0])

print(sim_lst)

from statistics import mean
cmp_mean = mean(sim_lst)

pred_algo = []
if cmp_mean >= 50:
    print('true')
elif cmp_mean < 50:
    print('fake')