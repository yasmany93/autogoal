import os
import pickle
from autogoal.contrib import find_classes
from autogoal.kb import Seq, Document, FeatureSet
from autogoal.ml import AutoML
from lda_model import LDAModel

#from cropsPlanningV2 import cropsFilter
#from epsilonGreedy import AlgorithmStructure

def load_documents(files_address="./data"):
    document_list = {}
    for file in os.listdir(files_address):
        if file.endswith(".txt"):
            with open(files_address + "/" + file, "rb") as address:
                document_list[file] = pickle.load(address)
    return document_list

def combine_text(list_of_documents):
    combined_texts = []
    for document in list_of_documents:
        combined_texts.append(' '.join(document))
    return combined_texts
            

if __name__ == "__main__":

    
    address = "./data"
    document_list = load_documents()
    texts = []
    for (k, v) in document_list.items():
        texts.append(v)
    
    combined_text = combine_text(texts)
    
    #Inicializo mi clase
    lda = LDAModel(combined_text)
    
    automl = AutoML(
        input=Seq[Document],
        output=FeatureSet,
        registry=[LDAModel] + find_classes()
    )

    res = {0: '0.016*"like" + 0.015*"right" + 0.014*"im" + 0.012*"dont" + 0.011*"know" + 0.011*"just" + 0.010*"fucking" + 0.007*"youre" + 0.007*"said" + 0.007*"went"', 
    1: '0.000*"like" + 0.000*"know" + 0.000*"dont" + 0.000*"just" + 0.000*"im" + 0.000*"right" + 0.000*"said" + 0.000*"shit" + 0.000*"got" + 0.000*"people"', 
    2: '0.035*"like" + 0.018*"im" + 0.017*"just" + 0.016*"know" + 0.014*"dont" + 0.010*"thats" + 0.009*"right" + 0.008*"youre" + 0.008*"people" + 0.007*"got"', 
    3: '0.000*"like" + 0.000*"im" + 0.000*"just" + 0.000*"know" + 0.000*"thats" + 0.000*"dont" + 0.000*"youre" + 0.000*"right" + 0.000*"got" + 0.000*"people"'}

    automl.fit(lda, res)
    
    score = automl.score(lda, res)

    print(score)
    #Ayado mi algoritmo a autoML
    
    # algorithmStructure mi clase inizializada
    # [1, 1, 1] salida de mi problema, ya conocida
    #automl.fit(lda, [1, 1, 1]) # Hago una corrida de AutoML para ver si encuentra mi algoritmo

    # Resultado de mi corrida de mi problema. Resultado entre 0 y 1
    

    



    # score = automl.
    # print(score)
    