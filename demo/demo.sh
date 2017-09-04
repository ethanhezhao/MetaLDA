#!/bin/bash

# compile MetaLDA

cd ..

mvn package

echo 'compiling finished ...'

cd demo


MALLET_LOCATION='../../Mallet-master' #change to your own Mallet location

dataset='WS'

# prepare training/testing documents

$MALLET_LOCATION/bin/mallet import-file --input ../data/$dataset/train_doc.txt \
--output ../data/$dataset/train_doc.mallet \
--label-as-features --keep-sequence --line-regex '([^\t]+)\t([^\t]+)\t(.*)' 

$MALLET_LOCATION/bin/mallet import-file --input ../data/$dataset/test_doc.txt \
--output ../data/$dataset/test_doc.mallet \
--label-as-features --keep-sequence --line-regex '([^\t]+)\t([^\t]+)\t(.*)' 

echo 'converting input documents finished ...'

# prepare word features

java -cp ../target/metalda-0.1-jar-with-dependencies.jar topicmodels.BinariseWordEmbeddings \
--train-docs ../data/$dataset/train_doc.mallet \
--test-docs ../data/$dataset/test_doc.mallet \
--input ../data/$dataset/raw_embeddings.txt \
--output ../data/$dataset/binary_embeddings.txt

echo 'binarising word embeddings finished ...'

# train MetaLDA

topics=100

alphamethod=1

betamethod=1

savedir=./save && mkdir -p $savedir;

java -Xmx4g -cp ../target/metalda-0.1-jar-with-dependencies.jar topicmodels.MetaLDATrain \
--train-docs ../data/$dataset/train_doc.mallet \
--num-topics $topics \
--word-features ../data/$dataset/binary_embeddings.txt \
--save-folder $savedir \
--sample-alpha-method $alphamethod \
--sample-beta-method $betamethod

echo 'training finished ...'


# inference without unseen words

java -Xmx4g -cp ../target/metalda-0.1-jar-with-dependencies.jar topicmodels.MetaLDAInfer \
--test-docs ../data/$dataset/test_doc.mallet \
--save-folder $savedir \
--compute-perplexity true;

echo 'inference without unseen words finished ...'

# inference with unseen words

java -Xmx4g -cp ../target/metalda-0.1-jar-with-dependencies.jar topicmodels.MetaLDAInferUnseen \
--test-docs ../data/$dataset/test_doc.mallet \
--save-folder $savedir \
--compute-perplexity true \
--word-features ../data/$dataset/binary_embeddings.txt 

echo 'inference with unseen words finished ...'
