import tiktoken


class Tokenizer:
    #class to tokenize and chunk text
    def __init__(self, max_tokens=500):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def txt_to_tokens(self, text):
        #tokenize text
        return self.encoding.encode(text)
    
    def create_chunks(self, text, max_tokens=500):
        #create chunks of text
        tokens = self.txt_to_tokens(text)
        chunks = []
        while len(tokens)>max_tokens:
            while self.encoding.decode([tokens[max_tokens]])!="." or max_tokens==0:
                max_tokens-=1
            chunks.append(tokens[:max_tokens+1])
            tokens = tokens[max_tokens:]
            max_tokens = 500
        chunks.append(tokens)   
        return chunks

