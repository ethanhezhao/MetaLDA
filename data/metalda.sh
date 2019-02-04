#!/bin/bash

#change to your own Mallet location
MALLET_LOCATION='/home/wbuntine/Code/mallet-2.0.8'
METALDA_LOCATION='/home/wbuntine/GitHub/MetaLDA'

#  this is a directory, must contain 'train_doc.txt'
dataset=$1
topics=$2

rm -rf $dataset/save

# prepare training/testing documents

$MALLET_LOCATION/bin/mallet import-file --input $dataset/train_doc.txt \
--output $dataset/train_doc.mallet \
--label-as-features --keep-sequence --line-regex '([^\t]+)\t([^\t]+)\t(.*)'

if [ -f $dataset/test_doc.txt ] ; then
    $MALLET_LOCATION/bin/mallet import-file --input $dataset/test_doc.txt \
                                --output $dataset/test_doc.mallet \
                                --label-as-features --keep-sequence --line-regex '([^\t]+)\t([^\t]+)\t(.*)'
fi

#  prepare embeddings

if [ ! -f $dataset/binary_embeddings.txt ] ; then
    if [ -f $dataset/test_doc.txt ] ; then
	java -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.BinariseWordEmbeddings --train-docs $dataset/train_doc.mallet --test-docs  $dataset/test_doc.mallet  --input $METALDA_LOCATION/data/raw_embeddings.txt --output $dataset/binary_embeddings.txt
    else
	java -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.BinariseWordEmbeddings --train-docs $dataset/train_doc.mallet --input $METALDA_LOCATION/data/raw_embeddings.txt --output $dataset/binary_embeddings.txt
    fi
fi

echo 'converting input documents finished ...'

# prepare word features

if [ -f $dataset/test_doc.txt ] ; then
    java -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.BinariseWordEmbeddings \
	 --train-docs $dataset/train_doc.mallet \
	 --test-docs $dataset/test_doc.mallet \
	 --input $dataset/raw_embeddings.txt \
	 --output $dataset/binary_embeddings.txt
else
    java -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.BinariseWordEmbeddings \
	 --train-docs $dataset/train_doc.mallet \
	 --input $dataset/raw_embeddings.txt \
	 --output $dataset/binary_embeddings.txt
fi

echo 'binarising word embeddings finished ...'

# train MetaLDA

alphamethod=1
betamethod=1
iter=20000
burn=5000

savedir=$dataset/save && mkdir -p $savedir;

java -Xmx4g -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.MetaLDATrain \
--train-docs $dataset/train_doc.mallet \
--num-topics $topics \
--num-iterations $iter \
--burn-in-period $burn \
--word-features $dataset/binary_embeddings.txt \
--save-folder $savedir \
--sample-alpha-method $alphamethod \
--sample-beta-method $betamethod

echo 'training finished ...'


# inference without unseen words
if [ -f $dataset/test_doc.txt ] ; then
    java -Xmx4g -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.MetaLDAInfer \
	 --test-docs $dataset/test_doc.mallet \
	 --save-folder $savedir \
	 --compute-perplexity true;
    
    echo 'inference without unseen words finished ...'
    
    # inference with unseen words
    
    java -Xmx4g -cp $METALDA_LOCATION/target/metalda-0.1-jar-with-dependencies.jar topicmodels.MetaLDAInferUnseen \
	 --test-docs $dataset/test_doc.mallet \
	 --save-folder $savedir \
	 --compute-perplexity true \
	 --word-features $dataset/binary_embeddings.txt 
    
    echo 'inference with unseen words finished ...'
fi
