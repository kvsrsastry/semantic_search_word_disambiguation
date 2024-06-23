#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
import argparse
import os
import sys
import json
from semantic_search_word_disambiguation import sswd

if __name__ == "__main__":
 args = argparse.ArgumentParser(prog='sswd_driver', description="Driver script to use semantic search word disambiguation package")
 args.add_argument('--version', '-v', action='version', version='%(prog)s 1.0')
 args.add_argument('--input-csv-file', '-i', type=str, required=True)
 args.add_argument('--create-vector-db', '-c', type=str)
 args.add_argument('--current-vector-db-path', '-d', type=str)
 args.add_argument('--create-ner-tags', '-n', type=str)
 args.add_argument('--current-ner-tags-path', '-t', type=str)
 args.add_argument('--input-sentence', '-s', type=str)
 arg = vars(args.parse_args())

 # One of the '--create-vector-db' and '--current-vector-db-path' arguments is mandatory
 if arg['create_vector_db'] is None and arg['current_vector_db_path'] is None:
  print("One of the '--create-vector-db' and '--current-vector-db-path' is mandatory. Please provide one of them.")
  sys.exit(1)
 
 # User should not input both the '--create-vector-db' and '--current-vector-db-path' arguments
 if arg['create_vector_db'] is not None and arg['current_vector_db_path'] is not None:
  print("Please input only one of the 2 arguments in '--create-vector-db' and '--current-vector-db-path'.")
  sys.exit(1)
 
 # One of the '--create-ner-tags' and '--current-ner-tags-path' arguments is mandatory
 if arg['create_ner_tags'] is None and arg['current_ner_tags_path'] is None:
  print("One of the '--create-ner-tags' and '--current-ner-tags-path' is mandatory. Please provide one of them.")
  sys.exit(1)
 
 # User should not input both the '--create-ner-tags' and '--current-ner-tags-path' arguments
 if arg['create_ner_tags'] is not None and arg['current_ner_tags_path'] is not None:
  print("Please input only one of the 2 arguments in '--create-ner-tags' and '--current-ner-tags-path'.")
  sys.exit(1)
 
 print(arg)
 
 dictionary_path = arg['input_csv_file']
 vector_database_path = arg['create_vector_db'] or arg['current_vector_db_path']
 ner_dictionary_path = arg['create_ner_tags'] or arg['current_ner_tags_path']

 # Instantiate the Class
 sswd_obj = sswd(dictionary_path, vector_database_path, ner_dictionary_path, arg['create_vector_db'] is not None, arg['create_ner_tags'] is not None)
 print(vars(sswd_obj))

 # Create vector database, if requested
 if arg['create_vector_db'] is not None:
  print("1. Creating Vector Database with input dictionary words and line_numbers as 'metadata' and word definitions as the text chunks for semantic similarity search.")
  print("2. This is a one time process and will take some time for the creation")
  sswd_obj.create_vector_database()
  print("FAISS Vector database with embeddings created successfully !!!")

 if arg['create_ner_tags'] is not None:
  print("1. Creating NER Tags, root_words, POS Tags for the given input vocabulary.")
  print("2. This is a one time process and will take some time for the creation as we invoke 'NER Transformer Pipeline' for all the Vocab words and definitions.")
  sswd_obj.create_ner_tags()
  print("NER Tags, root_words, POS Tags for the given input vocabulary created successfully !!!")

 if arg['input_sentence'] is not None:
  output_dict = sswd_obj.parse_and_disambiguate_words(arg['input_sentence'])
  print(json.dumps(output_dict, indent=2))

