import tokenisation as tk
import search as s
from fastapi import FastAPI, HTTPException
import uvicorn


def main():
    # This is the main function
    with open(".env", 'r') as fichier:
        # Lisez le contenu du fichier et attribuez-le à la variable cle_api
        cle_api = fichier.read().strip()
    
    tokenizer=tk.Tokenizer()
    input_text = open("1_transcript.txt", "r",encoding='utf-8').read()
    chunks = tokenizer.create_chunks(text=input_text, max_tokens=500)
    qa_engine = s.QAVecLLM(chunks,cle_api)

    question = "Quelle est la réponse dans le deuxième chunk ?"
    answer = qa_engine.find_answer(question)
    print(f"Question : {question}\nRéponse : {answer}")
    
    # app=FastAPI()
    # @app.get("/")
    # def read_root():
    #     return {"message": "Bienvenue dans l'API de questions-réponses"}

    # @app.get("/answer/")
    # def get_answer(question: str):
    #     if not question:
    #         raise HTTPException(status_code=400, detail="La question est requise")
    #     a=answer 
    #     return {"question": question, "answer": a}

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    

if __name__ == "__main__":
    main()