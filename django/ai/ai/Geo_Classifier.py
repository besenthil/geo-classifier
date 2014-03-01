import nltk
from   urllib   import urlopen
from   nltk.tokenize.punkt import PunktWordTokenizer
from   nltk import word_tokenize, wordpunct_tokenize
import sys
from   nltk.corpus.reader import WordListCorpusReader
import pickle

class Geo_Classifier:
    ''' Classifies a given wikipedia URL into a STATE or a CITY'''
    # NLTK DATA home path. This is where the training data set is present.
    TRAINED_FILE = '/app/ai/train_file.txt'
    _CLASSIFIER_ACCURACY = 0.6

    def normalize_word(self, word):
        # Remove unnecessary "spam" characters

        return filter(lambda word: word not in 'the,"inisarefromwithonfor1234567890asbywasretrieved.andof:;''()).{}[]-to\'&#^%160/20102011201220132009ateditwikipedia].isbn\x80\x93),.&#\xe0\xe1has\xd0\xd1\xb0worldthisthat|&amp201420072006200520042003200220012000+=-', word)

    def extract_geo_features (self,word):
        # Feature extractor.
        return {'any_word':word}

    def geo_train_corpus(self):
        with open(Geo_Classifier.TRAINED_FILE, 'r') as f:
            l_list = pickle.load(f)
        self.geo_classifier = nltk.NaiveBayesClassifier.train(l_list)
        return self.geo_classifier

    def clean_url(self,url):
        self.raw_url_cleansed = nltk.clean_html(urlopen(url).read())
        return self.raw_url_cleansed

    def return_message(self,url,result,prob):
        return ('Geo Classifier classifies the URL: ' + url + ' as a ' + result + ' with a probability of ' + str(prob*100))  
   
    def __init__(self,url):
        self.l_pdiststate, self.l_pdistcity = [],[]
     
    def geo_classify(self,url):
        self.tokens=nltk.FreqDist(self.normalize_word(word.lower() for word in wordpunct_tokenize(self.clean_url(url))))
 
        for word in self.tokens.keys()[:5]:
            self.pdiststate,self.pdistcity=self.geo_classifier.prob_classify(self.extract_geo_features(word)).prob('state'),self.geo_classifier.prob_classify(self.extract_geo_features(word)).prob('city')
            self.l_pdiststate.append(self.pdiststate)
            self.l_pdistcity.append(self.pdistcity)
        
        if(max(self.l_pdiststate) > max(self.l_pdistcity)) and max(self.l_pdiststate) > Geo_Classifier._CLASSIFIER_ACCURACY:    
            return (self.return_message(url,'State',max(self.l_pdiststate)))
        elif (max(self.l_pdistcity) > max(self.l_pdiststate)) and max(self.l_pdistcity) > Geo_Classifier._CLASSIFIER_ACCURACY:
            return (self.return_message(url,'City',max(self.l_pdistcity)))
        else:
            return(self.return_message(url,'Not a City or State',max(self.l_pdistcity)))
'''
if __name__ == "__main__":
      url=sys.argv[1]
      print url
      g=Geo_Classifier(url)
      g.geo_train_corpus()
      #print type(g)
      print g.geo_classify(url)
'''
