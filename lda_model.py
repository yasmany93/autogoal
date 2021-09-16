import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils, models
import scipy.sparse
from nltk import word_tokenize, pos_tag
import abc
from autogoal.kb import SemanticType


class Algorithm(SemanticType):
    pass

class LDAStructure(Algorithm):
    # Metodo abstracto que implementa LDAModel
    @abc.abstractmethod
    def list_to_dict(self):
        pass
    # Redifinir _match para que machee con los que hereden de LDAAbstract
    @classmethod
    def _match(cls, x) -> bool:
        return issubclass(x, LDAAbstract)

class LDAAbstract(abc.ABC):
    pass

class LDAModel(LDAAbstract):

    def __init__(self, texts_content=[]):
        
        self.texts_content = texts_content
        self.clean_data()
        self.term_document_matrix()
        #self = PreProcessingData(texts_content)
        
        def tdm_to_corpus(tdm):
            sparse_counts = scipy.sparse.csr_matrix(tdm)
            corpus = matutils.Sparse2Corpus(sparse_counts)
            return corpus

        self.corpus = tdm_to_corpus(self.tdm)
        
        id2word = dict((v, k) for k, v in self.cv.vocabulary_.items())
        self.i2w = id2word
        
        self.topics = 4
        self.iterations = 50

        self._lda_ = models.LdaModel(corpus=self.corpus, id2word=self.i2w, num_topics=self.topics, passes=self.iterations)

        '''  
        def document_corresponding_topic(_lda_, corpus):
            corpus_transformed = _lda_[corpus]
            data_dtm = self.tdm.transpose()
            topic_document = list(zip([a for [(a,b)] in corpus_transformed], data_dtm.index))
            return topic_document

        self.topic_doc = document_corresponding_topic(self._lda_, self.corpus)
        '''
        print(self.list_to_dict())
        

    def clean_data(self):

            self.document_list = {}
            for i in range(len(self.texts_content)):
                self.document_list[i.__str__() + "_text"] = [self.texts_content[i]]
            
            #convertir de diccionario a dataframe de pandas
            pd.set_option('max_colwidth',200) #para revisar datos
            self.dataframe = pd.DataFrame.from_dict(self.document_list).transpose()
            self.dataframe.columns = ['text']
            self.dataframe = self.dataframe.sort_index()
            
            def cleaning_techniques(text):
                text = text.lower() #llevar todo a minuscula
                text = re.sub('\[.*?\]', '', text) #quitar text entre corchetes
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text) #quitar signos de puntuacion
                text = re.sub('\w*\d\w*', '', text) #quitar palabras que contengan numeros
                text = re.sub('[‘’“”…]', '', text) #quitar comillas
                text = re.sub('\n', '', text) #quitar caracter \n
                

                def nouns_adj(_text):
                    #Del texto dejo los sustantivos y adjetivos porque la distribucion de palabras con los temas
                    #no tiene mucho sentido de sin hacer este filtro
                    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
                    tokenized = word_tokenize(_text)
                    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
                    return ' '.join(nouns_adj)
            
                nouns_adj(text)
                return text

            self.dataframe = pd.DataFrame(self.dataframe.text.apply(cleaning_techniques))
        
    def term_document_matrix(self):
            self.cv = CountVectorizer(stop_words='english')
            data_cv = self.cv.fit_transform(self.dataframe.text)
            data_dtm = pd.DataFrame(data_cv.toarray(), columns=self.cv.get_feature_names())
            data_dtm.index = self.dataframe.index
            self.tdm = data_dtm.transpose()
    

    def list_to_dict(self):
        lda_results = {}
        res = self._lda_.print_topics()

        for i in range(len(res)):
            k = res[i][0]
            v = res[i][1]
            lda_results[k] = v 

        return lda_results
        



