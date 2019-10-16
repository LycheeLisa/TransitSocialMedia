import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
import re
import time
import gensim
from gensim.models import CoherenceModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, rand
import numpy as np
from selenium import webdriver
from haversine import haversine
import numbers

def mindist(dist_list,station_list):
    min_dist=1000000
    min_case=np.nan
    for i in range(0,len(dist_list)-1):
        if dist_list[i] < min_dist:
            min_dist=dist_list[i]
            min_case=station_list[i]
    if min_dist==1000000:
        min_dist=np.nan
    return min_dist,min_case 

def find_lat_long(address_list, patient_time):
    lat_long_list=list()
    for address in address_list:
        try:
            driver = webdriver.Chrome()
            driver.get('https://getlatlong.net')
            time.sleep(patient_time)
            address_box = driver.find_element_by_id("addr")
            address_box.clear()
            address_box.send_keys(address)
            go_button = driver.find_element_by_xpath('//*[@id="o"]/table/tbody/tr/td[1]/div[1]/table/tbody/tr/td[2]/input[2]')
            go_button.click()
            time.sleep(patient_time)
            lat_box_content = driver.find_element_by_id("latbox")
            long_box_content = driver.find_element_by_id("lonbox")
            lat_long_list.append([lat_box_content.get_attribute('value'),long_box_content.get_attribute('value')])
            print("{0} : lat={1} and long={2}".format(address,lat_box_content.get_attribute('value'),long_box_content.get_attribute('value')))
            time.sleep(patient_time)
            driver.close()
        except:
            lat_long_list.append([0.0,0.0])
            print("Error: {0} : lat={1} and long={2}".format(address,0.0,0.0))
            driver.close()
    return lat_long_list

def distance_func(location, location_list):
    distance_list=list()
    if isinstance(location[0], numbers.Number) & isinstance(location[1], numbers.Number):
        for loc in location_list:
            if isinstance(loc[0], numbers.Number) & isinstance(loc[1], numbers.Number):
                distance=haversine(location, loc, unit='km')
                distance_list.append(distance)
            else:
                distance_list.append(None)
    else:
        distance_list=[None]*len(location_list)
    return distance_list

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
        lemma_str = [token.lemma_ for token in input_str if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]
        filtered_str = [w for w in lemma_str if not w in self.stop_words]
        return [filtered_str]


class gensim_optimizer:
    def __init__(self,model_name, model_path, dictionary, corpus, texts, max_evals):
        self.model_name = model_name
        self.model_path=model_path
        self.dictionary = dictionary
        self.corpus = corpus
        self.texts = texts
        self.max_evals=max_evals

        self.hyper_parameters=None
        self.best_score=0
        self.best_model = None
        self.trails=None

        self.set_hyper_parameters()

        pass

    def set_hyper_parameters(self):
        if self.model_name.lower() == "ldamodel":
            # 'distributed': hp.choice('distributed', [True, False])
            # 'chunksize': hp.quniform('chunksize', 10000, 5000, 100000)
            # 'gamma_threshold':hp.loguniform('gamma_threshold', -3, 2),
            # 'minimum_phi_value':hp.loguniform('minimum_phi_value', -3, 2)
            self.hyper_parameters={'num_topics': hp.quniform('num_topics', 1, 50, 1),
                                   'passes': hp.quniform('passes', 3, 30, 1),
                                   'decay': hp.uniform('decay', 0.5, 1),
                                   'alpha': hp.choice('alpha', ["asymmetric", "auto"])}

        if self.model_name.lower() == "ldamallet":
            # 'optimize_interval': hp.quniform('optimize_interval', 1, 50, 1)
            self.hyper_parameters={'num_topics': hp.quniform('num_topics', 1, 200, 1),
                                   'alpha': hp.quniform('alpha', 3, 200, 1),
                                   'topic_threshold': hp.uniform('topic_threshold', 0.001, 1)}

    def exceute(self):
        trials = Trials()
        best = fmin(lambda x: self.bayesian_optimizer(x),
                    self.hyper_parameters,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    trials=trials)

        return trials, self.best_model, self.best_score

    def bayesian_optimizer(self, xx):
        coherence_values = -self.compute_coherence_values(xx)
        if -coherence_values > self.best_score:
            self.best_score = -coherence_values

        return {'loss': coherence_values, 'status': STATUS_OK}

    def compute_coherence_values(self, x):
        if self.model_name.lower() == "ldamodel":
            x['num_topics'] = int(x['num_topics'])
            x['passes'] = int(x['passes'])
            model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                    id2word=self.dictionary,
                                                    random_state=400,
                                                    **x)
            # coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')

        if self.model_name.lower() == "ldamallet":
            x['num_topics'] = int(x['num_topics'])
            x['alpha'] = int(x['alpha'])
            model = gensim.models.wrappers.LdaMallet(self.model_path,
                                                     corpus=self.corpus,
                                                     id2word=self.dictionary,
                                                     random_seed=400,
                                                     **x)
        coherencemodel = CoherenceModel(model=model,
                                        texts=self.texts,
                                        dictionary=self.dictionary,
                                        coherence='c_v')

        coherence_score = coherencemodel.get_coherence()
        if coherence_score > self.best_score:
            self.best_model = model
            self.best_score = coherence_score

        return coherence_score
 
class mp_address_latlong:
    def __init__(self, address_list, patient_time, jobs):
        self.address_list=address_list
        self.patient_time=patient_time
        self.jobs=jobs

        pass

    def excecute(self):
        p=mp.Pool(processes=self.jobs)
        results=p.map(self.webscraper,self.address_list)
        p.close()
        p.join()
        return list(results)

    def webscraper(self, one_address):
        try:
            try:
                driver = webdriver.Chrome()
                driver.get('https://getlatlong.net')
                time.sleep(self.patient_time)
                address_box = driver.find_element_by_id("addr")
                address_box.clear()
                address_box.send_keys(one_address)
                go_button = driver.find_element_by_xpath('//*[@id="o"]/table/tbody/tr/td[1]/div[1]/table/tbody/tr/td[2]/input[2]')
                go_button.click()
                time.sleep(self.patient_time)
                lat_box_content = driver.find_element_by_id("latbox")
                long_box_content = driver.find_element_by_id("lonbox")
                lat_long=[lat_box_content.get_attribute('value'),long_box_content.get_attribute('value')]
#                print("{0} : lat={1} and long={2}".format(one_address,lat_box_content.get_attribute('value').encode('utf8'),long_box_content.get_attribute('value').encode('utf8')))
                time.sleep(self.patient_time)
                driver.close()
            except:
                lat_long=["0.0","0.0"]
#                print("Error: {0} : lat={1} and long={2}".format(one_address,0.0,0.0))
                driver.close()
        except:
            lat_long=["0.0","0.0"]
#            print("Error: {0} : lat={1} and long={2}".format(one_address,0.0,0.0))
            
        return lat_long

        