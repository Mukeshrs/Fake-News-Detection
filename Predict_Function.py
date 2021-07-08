import requests
import bs4
import pickle
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from statistics import mean
from statistics import mode
stopwords = stopwords.words('english')


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def cosine_sim_vectors(vec1,vec2):
    vec1 = vec1.reshape(1,-1)
    vec2 = vec2.reshape(1,-1)
    return cosine_similarity(vec1,vec2)

def predict_function(message):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('vectorizer_ml.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('classification_ml.model', 'rb'))

    # make a prediction
    ml_pred = loaded_model.predict(loaded_vectorizer.transform([message]))
    #print(loaded_model.predict(loaded_vectorizer.transform([utt])))

    # load the vectorizer
    loaded_vectorizer = pickle.load(open('vectorizer_dl.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('classification_dl.model', 'rb'))

    # make a prediction
    dl_pred = loaded_model.predict(loaded_vectorizer.transform([message]))
    #print(loaded_model.predict(loaded_vectorizer.transform([utt])))

    # Make two strings with default google search URL
    # 'https://google.com/search?q=' and
    # our customized search keyword.
    # Concatenate them
    text = "India plans on going to war with USA"
    url = 'https://google.com/search?q=news' + text

    # Fetch the URL data using requests.get(url),
    # store it in a variable, request_result.
    request_result = requests.get(url)

    # Creating soup from the fetched request
    soup = bs4.BeautifulSoup(request_result.text,
                             "html.parser")
    # print(soup)

    # soup.find.all( h3 ) to grab
    # all major headings of our search result,
    heading_object = soup.find_all('h3')

    # Iterate through the object
    # and print it as a string.
    lst = []
    lst.append(text)

    for info in heading_object:
        # print(info.getText())
        lst.append(info.getText())

    cleaned = list(map(clean_string, lst))
    #print(cleaned)

    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()

    cmp = vectors[0]
    sim_lst = []

    for i in range(1, len(vectors)):
        cal = cosine_sim_vectors(cmp, vectors[i])
        sim_lst.append(cal[0][0])

    Final_Pred = []
    cmp_mean = mean(sim_lst)

    if cmp_mean >= 50:
        print('true')
        Final_Pred.append('true')
    elif cmp_mean < 50:
        print('fake')
        Final_Pred.append('fake')

    Final_Pred.append(ml_pred[0])
    Final_Pred.append(dl_pred[0])

    pred= mode(Final_Pred)

    return pred


#predict('India plans on going to war with USA')