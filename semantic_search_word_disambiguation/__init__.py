#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
from transformers import pipeline
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer
import pandas as pd
import numpy as np
import requests
import re
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class sswd:
 def __init__(self, dictionary_path, vector_database_path, ner_dictionary_path, is_create_db, is_create_ner):
  self.dictionary_path = dictionary_path
  self.vector_database_path = vector_database_path
  self.ner_dictionary_path = ner_dictionary_path
  self.ner = pipeline("ner")
  self.lemmatizer = WordNetLemmatizer()
  if not is_create_ner:
    self.udict = self.parse_updated_dictionary_csv(self.ner_dictionary_path)
  self.ner_map = self.get_ner_map()
  self.hf = self.get_hugging_face_embedding()
  if not is_create_db:
    self.local_vector_store = self.load_vector_store()

 def get_hugging_face_embedding(self):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf

 def parse_input_file(self, input_file):
    # Check if the input file exists
    if not os.path.exists(input_file):
      print("No such file: File '{}' does not exist.".format(input_file))
      return None
    
    # Check if the input file is a file
    if not os.path.isfile(input_file):
      print("Given input '{}' is not a file. Please provide a filename.".format(input_file))
      return None
    
    # Check if the input file is a csv
    df = None
    try:
      df = pd.read_csv(input_file)
    except Exception as e:
      print("Given input file '{}' is either not a csv (or) it has some issues. Please fix them".format(input_file))
      print(e)
      return None

    # Input file exists and is a valid csv file
    words = df['word'].tolist()
    definitions = df['definition'].tolist()
    # Check if the number of words equal to number of definitions
    if not len(words) == len(definitions):
      print("In the given input file '{}', #words not matching #definitions. Please fix the issue".format(input_file))
      return None
    
    # Eliminate duplicate definitions of the same word
    # For now, don't eliminate duplicates
    dedup_words = []
    dedup_definitions = []
    tmpd = dict()
    for wrd, defi in zip(words, definitions):
      key = '{} {}'.format(' '.join(wrd.strip().lower().split()), ' '.join(defi.strip().lower().split()))
      if key not in tmpd:
        #tmpd[key] = 1
        dedup_words.append(wrd)
        dedup_definitions.append(defi)
    return dedup_words, dedup_definitions

 def split_definitions_into_chunks(self, words, definitions):
    vs_words = []
    vs_definitions = []
    CHUNK_SIZE = 512
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_SIZE//10)
    for ind, (word, definition) in enumerate(zip(words, definitions)):
      tmp_chunks = splitter.split_text(definition)
      tmp_words = [{"source": word, "line_num": ind+2}] * len(tmp_chunks)
      vs_words.extend(tmp_words)
      vs_definitions.extend(tmp_chunks)
    return vs_words, vs_definitions

 def create_vector_database_helper(self, vs_words, vs_definitions):
  vector_store = FAISS.from_texts(texts=vs_definitions, metadatas=vs_words, embedding=self.hf)
  vector_store.save_local(self.vector_database_path)

 def create_vector_database(self):
  words, definitions = self.parse_input_file(self.dictionary_path)
  vs_words, vs_definitions = self.split_definitions_into_chunks(words, definitions)
  self.create_vector_database_helper(vs_words, vs_definitions)
  self.local_vector_store = self.load_vector_store()

 def mark_ner_tags(self, words, definitions):
    ners = []
    root_words = []
    poss = []
    for wrd, defi in zip(words, definitions):
      cwrd = wrd.capitalize()
      lowercase_defi = defi.lower()
      sentence = '{} is {}'.format(cwrd, lowercase_defi)
      tmp_ner = None
      for d in self.ner(sentence):
        if d['start'] == 0:
          tmp_ner = d['entity']
          break
      ners.append(tmp_ner)
      root_words.append(self.lemmatizer.lemmatize(wrd.lower()))
      poss.append(nltk.pos_tag([wrd.lower()])[0][1])
    return ners, root_words, poss

 def create_ner_tags(self):
  words, definitions = self.parse_input_file(self.dictionary_path)
  ners, root_words, poss = self.mark_ner_tags(words, definitions)
  udf = pd.DataFrame({
    'word': words,
    'definition': definitions,
    'ner': ners,
    'root_word': root_words,
    'pos': poss
  })
  udf.to_csv(self.ner_dictionary_path, index=None)
  self.udict = self.parse_updated_dictionary_csv(self.ner_dictionary_path)

 def get_input_wrd_positions(self, sent):
   pos_dict = dict()
   ind = 0
   for wrd in sent.split():
     pos_dict[ind] = wrd
     ind = ind + len(wrd) + 1
   return pos_dict

 def convert_ner_output(self, lst):
   ner_dict = dict()
   for d in lst:
     ner_dict[d['start']] = { 'entity': d['entity'], 'word': d['word'] }
   return ner_dict

 def convert_pos_output(self, lst):
   pos_dict = dict()
   for wrd, tag in lst:
     if tag.startswith('N'):
       #tagm = 'Noun'
       tagm = wordnet.NOUN
     elif tag.startswith('V'):
       #tagm = 'Verb'
       tagm = wordnet.VERB
     elif tag.startswith('J'):
       #tagm = 'Adjective'
       tagm = wordnet.ADJ
     elif tag.startswith('R'):
       #tagm = 'Adverb'
       tagm = wordnet.ADV
     else:
       tagm = wordnet.NOUN
     pos_dict[wrd.lower()] = tagm
   return pos_dict

 def get_ner_map(self):
    ner_map = {
    'I-PER': 'B-PER',
    'B-PER': 'I-PER',
    'I-LOC': 'B-LOC',
    'B-LOC': 'I-LOC',
    'I-ORG': 'B-ORG',
    'B-ORG': 'I-ORG',
    'I-MISC': 'B-MISC',
    'B-MISC': 'I-MISC'
    }

    return ner_map
  
 def parse_updated_dictionary_csv(self, fname):
    udf = pd.read_csv(fname) 
    udict = dict()
    line_num = 2
    for arr in list(udf.iloc[:,:].values):
      w,d,n,r,p = list(arr)
      if w.lower() not in udict:
        udict[w.lower()] = list()
      udict[w.lower()].append({
            'word': w,
            'definition': d,
            'ner': n,
            'root_word': r,
            'pos': p,
            'line_num': line_num
      })
      line_num += 1
    return udict

 def load_vector_store(self):
  local_vector_store = FAISS.load_local(self.vector_database_path, self.hf, allow_dangerous_deserialization=True)
  return local_vector_store

 def parse_and_disambiguate_words(self, sent):
  input_sentence = sent.strip()
  input_sentence = ' '.join(input_sentence.split())
  input_wrd_positions = self.get_input_wrd_positions(input_sentence)

  ner_input_sentence = self.ner(input_sentence)
  input_ner_wrd_positions = self.convert_ner_output(ner_input_sentence)

  input_pos_tags = nltk.pos_tag(input_sentence.split())
  input_pos_tags_converted = self.convert_pos_output(input_pos_tags)

  #print(input_wrd_positions)
  #print(input_ner_wrd_positions)
  #print(input_pos_tags_converted)

  json_output = list()
  for k, v in input_wrd_positions.items():
    # As per NER, if the input word is an ORG, LOC, PER,
    # Take this flow
    json_output_d = dict()
    if k in input_ner_wrd_positions:
      entity = input_ner_wrd_positions[k]['entity']
      word = input_ner_wrd_positions[k]['word'].lower()
      json_output_d['word_start_position'] = k
      json_output_d['word'] = word
      # If the current word is present in the input dictionary
      if word in self.udict:
        for d in self.udict[word]:
          if entity == d['ner'] or self.ner_map[entity] == d['ner']:
            json_output_d['input_csv_line_number'] = str(d['line_num'])
            json_output_d['input_csv_related_word'] = d['word']
            break
        else:
          json_output_d['input_csv_line_number'] = 'NULL'

      else:
        json_output_d['input_csv_line_number'] = 'NULL'

    # If the word is not present in NER Output
    # Take this flow
    else:
      word = v.lower()
      json_output_d['word_start_position'] = k
      json_output_d['word'] = word
      # Present in dictionary
      if word in self.udict:
        tmp_scores = []
        for d in self.udict[word]:
          for l in self.local_vector_store.similarity_search_with_score(d['definition'], k=1):
            tmp_scores.append([l[1], l[0].metadata['source'], d['line_num']])
        tmp_scores.sort()
        if len(tmp_scores) > 0 and float(tmp_scores[0][0]) < 0.9:
          json_output_d['input_csv_line_number'] = str(tmp_scores[0][2])
          json_output_d['input_csv_related_word'] = tmp_scores[0][1]
          json_output_d['similarity_score'] = str(tmp_scores[0][0])
        else:
          json_output_d['input_csv_line_number'] = 'NULL'

      # Not present in dictionary
      # Case 1: Get the "synsets and their definitions from the wordnet" and "use the POS Tag structure"
      else:
        #word_meaning = dictionary.meaning(word)
        noun_verb = input_pos_tags_converted[word]
        syn = wordnet.synsets(word, pos=noun_verb)      
        tmp_scores = []
        for item in syn:
          for l in self.local_vector_store.similarity_search_with_score(item.definition(), k=1):
            tmp_scores.append([l[1], l[0].metadata['source'], l[0].metadata['line_num']])
        tmp_scores.sort()
        if len(tmp_scores) > 0 and float(tmp_scores[0][0]) < 0.9:
          json_output_d['input_csv_line_number'] = str(tmp_scores[0][2])
          json_output_d['input_csv_related_word'] = tmp_scores[0][1]
          json_output_d['similarity_score'] = str(tmp_scores[0][0])
        else:
          json_output_d['input_csv_line_number'] = 'NULL'
    json_output.append(json_output_d)

  return json_output