import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QAVecLLM:
    #Classe de recherhe de réponse dans une base de données vectorielle via GTP 3.5
    def __init__(self, chunks,key):

        self.documents = [' '.join(map(str, chunk)) for chunk in chunks]

        # Créer une base de données vectorielle TF-IDF à partir des documents
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(self.documents)

        # Initialise le modèle de langage (GPT-3 dans ce cas)
        self.openai_api_key = key
        openai.api_key = self.openai_api_key

    def find_answer(self, question):
        # Utilise GPT-3.5 pour générer une réponse initiale
        gpt3_response = openai.Completion.create(
            engine="davinci",
            prompt=question,
            max_tokens=100  
        )
        initial_answer = gpt3_response.choices[0].text.strip()

        # Utilise la base de données vectorielle pour trouver le document le plus similaire
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.document_vectors)
        
        # Sélectionne les indices des n=3 groupes de chunks les plus similaires
        n = 3
        most_similar_indices = np.argsort(similarities[0])[::-1][:n]

        # Utilise GPT-3.5 pour raffiner la réponse basée sur le document sélectionné
        refined_answers = []
        for idx in most_similar_indices:
            document = ' '.join(map(str, self.documents[idx]))  # Convertit les chunks en une seule chaîne
            gpt3_response = openai.Completion.create(
                engine="davinci",
                prompt=f"Question : {question}\nRéponse initiale : {initial_answer}\nDocument : {document}",
                max_tokens=100  
            )
            refined_answer = gpt3_response.choices[0].text.strip()
            refined_answers.append(refined_answer)

        return refined_answer
