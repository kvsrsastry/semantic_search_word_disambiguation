#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
#omgamganapathayenamaha
with open("test_sentences.txt") as fh:
 with open("tests.sh", "w") as fw:
  fw.write("#!/bin/sh\n\n")
  cnt = 1
  for line in fh:
   line = line.strip()
   fw.write('python3 /content/semantic_search_word_disambiguation/sswd_driver.py \
 --input-csv-file /content/semantic_search_word_disambiguation/vocabulary/Dictionary.csv \
 --current-vector-db-path /content/semantic_search_word_disambiguation/vector_db/faiss_index \
 --current-ner-tags-path /content/semantic_search_word_disambiguation/vocabulary/updated_dictionary.csv \
 --input-sentence {} \
 --output-json-file /content/semantic_search_word_disambiguation/test_script_and_results/test_output_json_files/output_{}.json'.format(line, cnt))
   fw.write("\n\n")
   cnt += 1

