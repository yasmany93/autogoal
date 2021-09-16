from autogoal.kb import Seq, AlgorithmBase, FeatureSet
from lda_model import LDAStructure

# Estructura necesaria para que AutoML encuentre tu clase
class TopicInference(AlgorithmBase):

    def __init__():
        pass

    #Metodo para que autoML encuentre el algoritmo. Dentro va toda la logica
    #de como utilizar el algoritmo y lo que devuelve
    def run(self, ldas: LDAStructure) -> FeatureSet:
        return ldas.list_to_dict()
