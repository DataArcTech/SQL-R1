import re
import json
import pdb

def read_json(file_path):
    with open(file_path, "r") as fin:
        raw_data = json.load(fin)
    return raw_data

class TokenPreprocessor:
    def __init__(self, 
                 snake_case=True, 
                 camel_case=True, 
                 normalize_text=True, 
                 replace_tokens=[], 
                 insert_space_when = ['.'],
                 tokenizer="T5"):
        
        self.snake_case = snake_case
        self.camel_case = camel_case
        self.normalize_text = normalize_text
        self.replace_tokens = replace_tokens
        self.insert_space_when = insert_space_when
        self.tokenizer = tokenizer
    
    def preprocess(self, query: str) -> str:
        if self.replace_tokens:
            query = self.replace_tokens_preprocessing(query)
        if self.camel_case:
            query = self.camel_case_preprocessing(query) # camel case before normalizing because normalizing will lowercase everything
        if self.normalize_text:
            query = self.normalize(query)
        if self.snake_case:
            query = self.snake_case_preprocessing(query)
        if self.insert_space_when:
            query = self.insert_space_preprocessing(query)
        return " ".join(query.split())
    
    def preprocess_all(self, data: list) -> list:
        refined_data = []
        for d in data:
            refined_data.append(self.preprocess(d))
        return refined_data
    
    def postprocess(self, query: str) -> str:
        if self.tokenizer == "T5":
            if self.insert_space_when:
                query = self.insert_space_preprocessing(query)
            if self.normalize_text:
                query = self.normalize(query)
            query = query.replace("average(", "avg (")
        if self.snake_case:
            query = self.snake_case_postprocessing(query)
        if self.camel_case:
            query = self.camel_case_postprocessing(query)
        if self.insert_space_when:
            query = self.insert_space_postprocessing(query)
        if self.replace_tokens:
            query = self.replace_tokens_postprocessing(query)
        return query
    
    def postprocess_all(self, data: list) -> list:
        refined_data = []
        for d in data:
            refined_data.append(self.postprocess(d))
        return refined_data
    
    def snake_case_preprocessing(self, query: str) -> str:
        query = query.split()
        for i, word in enumerate(query):
            if '_' in word and not word.endswith('_') and not word.startswith('_'):
                query[i] = word.replace('_', ' _ ')
        return ' '.join(query)
    
    def camel_case_preprocessing(self, query: str) -> str:
        return re.sub(r'([a-z])([A-Z])', r'\1## \2', query)
    
    def normalize(self, query: str) -> str:
        def white_space_fix(s):
            # Remove double and triple spaces
            return " ".join(s.split())

        def lower(s):
            # Convert everything except text between (single or double) quotation marks to lower case
            return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

        return white_space_fix(lower(query))
    
    def replace_tokens_preprocessing(self, query: str) -> str:    
        for token in self.replace_tokens:
            query = query.replace(token[0], token[1])
        return query
    
    def insert_space_preprocessing1(self, query: str) -> str:
        """Insert space before and after tokens in insert_space_when except when they are in quotes"""
        for token in self.insert_space_when:
            query = query.replace(token, ' ' + token + ' ')
        return query
    
    def insert_space_preprocessing(self, query: str)-> str:
        """Insert space before and after tokens in insert_space_when except when they are in quotes"""
        query = query.split()
        inside_quotes = False
        for i, word in enumerate(query):
            if word.startswith('"') or word.startswith("'") or word.startswith("\""):
                inside_quotes = not inside_quotes
            for token in self.insert_space_when:
                if token in word and not inside_quotes:
                    query[i] = word.replace(token, ' ' + token + ' ')
        return ' '.join(query)
    
    def snake_case_postprocessing(self, query: str) -> str:
        return query.replace(' _ ', '_')
    
    def camel_case_postprocessing(self, query: str) -> str:
        query = query.split()
        for i, word in enumerate(query):
            if '##' in word:
                query[i+1] = query[i+1].capitalize()
        query = ' '.join(query)
        return query.replace('## ', '')
    
    def replace_tokens_postprocessing(self, query: str) -> str:
        for token in self.replace_tokens:
            query = query.replace(token[1], token[0])
        return query
    
    def insert_space_postprocessing1(self, query: str) -> str:
        for token in self.insert_space_when:
            query = query.replace(' ' + token + ' ', token)
        return query
    
    def insert_space_postprocessing(self, query: str) -> str:
        """Remvoe space before and after tokens in insert_space_when except when they are in quotes"""
        query = query.split()
        inside_quotes = False
        for i, word in enumerate(query):
            if word.startswith('"') or word.startswith("'") or word.startswith("\""):
                inside_quotes = not inside_quotes
            for token in self.insert_space_when:
                if token == word and not inside_quotes:
                    query[i] = word.replace(token, '###' + token + '###')
        return ' '.join(query).replace('### ', '').replace(' ###', '')
    
    def test(self, query: str) -> str:
        query1 = self.normalize(query) # normalize 
        query2 = self.postprocess(self.preprocess(query))
        assert query1.lower() == query2.lower()
    
    def test_all(self, data: list) -> None:
        for td in data:
            try:
                self.test(td)
            except AssertionError:
                print(self.normalize(td))
                print(self.postprocess(self.preprocess(td)))
                pdb.set_trace()

def main():
    data = read_json('data/natsql/train.json')
    queries = [d['query'] for d in data]
    replace_tokens = [(' avg (', ' average ('), (' asc ', ' ascending '), (' desc ', ' descending ')]
    token_preprocessor = TokenPreprocessor(snake_case=True, camel_case=True, normalize_text=True, replace_tokens=replace_tokens, insert_space_when=['.'])
    token_preprocessor.test_all(queries)

if __name__ == "__main__":
    main()
    

    