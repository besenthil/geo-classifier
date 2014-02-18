import nltk
from   urllib   import urlopen
from   nltk.tokenize.punkt import PunktWordTokenizer
from   nltk import word_tokenize, wordpunct_tokenize
import sys
from   nltk.corpus.reader import WordListCorpusReader
import pickle

# Remove unnecessary "spam" characters
def word_normalizer(word):
    return filter(lambda word: word not in 'the,"inisarefromwithonfor1234567890asbywasretrieved.andof:;''()).{}[]-to\'&#^%160/20102011201220132009ateditwikipedia].isbn\x80\x93),.&#\xe0\xe1has\xd0\xd1\xb0worldthisthat|&amp201420072006200520042003200220012000+=-', word)

# Feature extractor
# Frequency count
def geo_features (word):
    return {'any_word':word}

# Initialize constants
NLTK_HOME = '/home/administrator/nltk_data'

l_list = []
# cleaning, tokenizing, normalizing

# Read the Corpus
state_reader = WordListCorpusReader(NLTK_HOME, ['state_files.txt'])
city_reader = WordListCorpusReader(NLTK_HOME, ['city_files.txt'])
train_file = '/app/ai/train_file.txt'
test_results_file = '/app/ai/test_city_results_file.txt'


# Store the URLs in  a list
urls = ([(url,'city') for url in city_reader.words()]+
        [(url,'state') for url in state_reader.words()]
        )

for url in list(urls):
    # Remove HTMLtabs after reading the URL
    raw = nltk.clean_html(urlopen(url[0]).read())
    print 'Finished cleaning html for ', url[0]
    # Compute the frequency distribution of the words
    tokens=nltk.FreqDist(word_normalizer(word.lower() for word in wordpunct_tokenize(raw)))
    print 'Finished computing FD for ', url[0]
    l_list = l_list + [(geo_features(word),url[1]) for word in tokens.keys()[:10]]
    print 'Finished extracting feature for ', url[0]

with open(train_file, 'w') as f:
    pickle.dump(l_list, f)

print 'Finished writing features into file ', train_file

with open(train_file, 'r') as f:
    l_list = pickle.load(f)

f.close()

print 'Finished reading geo features from file ', train_file

classifier = nltk.NaiveBayesClassifier.train(l_list)

print 'Finished training Naive Bayes Model on the geo features'

state_test_reader = WordListCorpusReader(NLTK_HOME, ['state_test_files.txt'])

print 'Finished reading the test dataset URL : ' , sys.argv[1]

print 'Starting to classify URL into either a CITY (or) a STATE'

with open(test_results_file, 'w') as f:

    for url in state_test_reader.words():
        raw = nltk.clean_html(urlopen(url).read())
        tokens=nltk.FreqDist(word_normalizer(word.lower() for word in wordpunct_tokenize(raw)))
        l_pdiststate, l_pdistcity = [],[]
        for word in tokens.keys()[:5]:
            pdiststate,pdistcity=classifier.prob_classify(geo_features(word)).prob('state'),classifier.prob_classify(geo_features(word)).prob('city')
            #print '%10s %8.5f %8.5f' % (word, pdiststate,pdistcity)
            l_pdiststate.append(pdiststate)
            l_pdistcity.append(pdistcity)
            #pdist=classifier.prob_classify(geo_features(word))
            #print '%10s %8.5f' % (word, pdist.prob(word))
        if(max(l_pdiststate) > max(l_pdistcity)) and max(l_pdiststate) > 0.6 :    
            f.write(url + '|State|' + str(max(l_pdiststate))+'\n')
        elif (max(l_pdistcity) > max(l_pdiststate)) and max(l_pdistcity) > 0.6:
            f.write(url + '|City|' +  str(max(l_pdistcity))+'\n')
        else:
            f.write(url + '|Not a City/State|' +  str(max(l_pdistcity))+'\n')

f.close()
