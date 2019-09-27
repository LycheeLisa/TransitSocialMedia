import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
import re

def flatten(x):
    """
    Function to flatten out nested list

    Parameters:
    ----------
    x : nested list

    Return:
    ----------
    [list elements removed from nested list]
    """
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

class mp_tokenize:
    def __init__(self, df, target_column, stop_words, station, photo, nlp,jobs):
        self.df=df
        self.target_column=target_column
        self.stop_words=stop_words
        self.station=station
        self.photo=photo
        self.nlp=nlp
        self.jobs = jobs
        pass

    def excecute(self):
        list_to_run=self.df[self.target_column]
        p=mp.Pool(processes=self.jobs)
        results=p.map(self.tokenize,list_to_run)
        p.close()
        p.join()
        return list(results)

    def tokenize(self, text):
        """
        Function to pre-process string

        Parameters:
        ----------
        text : comment string
        Return:
        ----------
        [processed string, [list of keywords]]
        """
        ### 1. Masking common strings
        if 'https://' in text:
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'urllink', text, flags=re.MULTILINE)
        processed_text = re.sub('[^A-Za-z]+', ' ', text).lower()
        processed_text = self.station.sub("ttcstation", processed_text)
        processed_text = self.photo.sub("photo", processed_text)
        ### 2. Get Lemma and conduct POS tagging
        input_str = self.nlp(processed_text)
        lemma_str = [token.lemma_ for token in input_str]
        filtered_str = ' '.join([w for w in lemma_str if not w in self.stop_words])
        return [filtered_str, self.get_keywords(input_str)]

    def get_keywords(self, text):
        """
        Function to extract chunks of key nouns and verbs

        Parameters:
        ----------
        text : comment string

        Return:
        ----------
        [list of unigram keywords ]
        """
        main_phrases = []
        for chunk in text.noun_chunks:
            if chunk.root.dep_ == 'nsubj' or chunk.root.dep_ == 'dobj' or chunk.root.dep_ == 'pobj':
                main_phrases.append(chunk.lemma_)
        for word in text:
            if word.pos_ == 'VERB':
                main_phrases.append(word.lemma_)
        final_phrases = flatten([i.split(' ') for i in main_phrases])
        return [w for w in final_phrases if w not in self.stop_words and '-PRON-' not in w]