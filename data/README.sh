
#  example use of scripts

# data directory
#  must have embeddings and data all set up
#     e.g. binary_embeddings.txt
DIR='WS'

# run metalda, second arg is number of topics
#  must have 'raw_embeddings.txt' available in top directory ..
#  usually its a link to a big 6Gb embeddings file somewhere else
#  first task of script is to extract the subset of embeddings needed ..
./metalda.sh WS 10

# use matlab to create diagnostics
# careful, all the quotes and frmatting must be exactly reproduced
#  note the directory ('WS') appears twice,
#  once to show where source docs are, another to show where results file is
matlab  -nodisplay -r 'read_stats("WA","WS/save"); exit';

