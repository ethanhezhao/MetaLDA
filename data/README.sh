
#  example use of scripts

# data directory
#  must have embeddings and data all set up
#     e.g. binary_embeddings.txt
DIR='WS'

# run metalda
./metalda.sh $DIR $N

# use matlab to create diagnostics
matlab  -nodisplay -r 'read_stats("'$DIR'"); exit';

