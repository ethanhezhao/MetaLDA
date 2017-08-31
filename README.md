# MetaLDA

# Basic Information
MetaLDA is a topic model that leverages either document or word meta information, or both of them jointly.

Key features:
1. Incorporates both document and word meta information in binary format
2. Implemented on top of [Mallet](http://mallet.cs.umass.edu) by JAVA
- Works with Mallet input format
- Runs efficiently (bit-coding and SparseLDA framework in Mallet apply)
- Runs with multi-threads (DistributedLDA framework in Mallet applies) 

# Run MetaLDA
1. Clone the repository or download the code
2. Compile the code with [Maven](https://maven.apache.org/what-is-maven.html):
- ```cd <metalda_location>```
- ```mvn package```
3. Prepare documents:
	- All documents (training/testing) are in Mallet's [LabeledLDA](http://www.mimno.org/articles/labelsandpatterns/) format. 
	- If the input documents are already in Mallet, they can be directly fed into the model. 
	- Otherwise, the documents have to be first converted into Mallet format. 
		- Each raw document should in the following format:
		```DOC_ID\tLABEL1 LABEL2 LABEL3\tWORD1 WORD2 WORD3\n```.  
		- Install [Mallet](http://mallet.cs.umass.edu) then use:
		```<mallet_location>/bin/mallet import-file --input <training/testing_doc_location> --output 	<training/testing_doc_mallet_location> --label-as-features --keep-sequence --line-regex '([^\t]+)\t([^\t]+)\t(.*)'```
4. Prepare word features:
	- MetaLDA uses the following sparse representation of binary word features:
	```WORD\tNNZ_INDEX1 NNZ_INDEX2 NNZ_INDEX3```
	- Use embeddings as word features
		- MetaLDA offers a function to binarise and convert word embeddings into the required word feature format. The raw input word embeddings are expected to follow the format of [GloVe](https://nlp.stanford.edu/projects/glove/):
 ```WORD EMBEDDING1 EMBEDDING2 EMBEDDING3```
 		- To binarise and convert the raw word embeddings, in the root folder of MetaLDA, use:
		```java -cp ./target/metalda-0.1-jar-with-dependencies.jar hezhao.BinariseWordEmbeddings --train-docs <training_doc_mallet_location>  --test-docs <testing_doc_mallet_location> --input <raw_embedding_location> --output <binary_embedding_location>```
 		- The function first reads the vocabularies of the training and testing documents (both in Mallet format) and then binarise the embeddings of the words in the vocabularies stored in the word embedding file, and finally saves the binarised embeddings into the required format. Note that MetaLDA does not require all the words in the training and testing documents have embeddings.
		
5. Train MetaLDA:

```java -cp ./target/metalda-0.1-jar-with-dependencies.jar hezhao.MetaLDATrain --train-docs <training_doc_mallet_location> --num-topics <num_topic> --word-features <binary_embedding_location> --save-folder <save_folder> --sample-alpha-method <sample_alpha_method> --sample-beta-method <sample_beta_method>```
- ```<sample_alpha_method>```: 
	- 0: fixed on initial value
	- 1: alpha is a full matrix sampled with doc labels
	- 2: alpha is sampled as an asymmetric vector over topic
	- 3: alpha is sampled as a single value
	- 4: alpha is an asymmetric vector over topics sampled by Mallet built-in sampler
	- 5: alpha is sampled as a single value by Mallet built-in sampler
- ```<sample_beta_method>```: 
	- 0: fixed on initial value
	- 1: beta is a full matrix sampled with word features
	- 2: beta is sampled as an asymmetric vector over topics
	- 3: beta is sampled as a single value
	- 4: beta is sampled as a single value by Mallet built-in sampler
- For more details of the arguments, please type in 
```java -cp ./target/metalda-0.1-jar-with-dependencies.jar --help```
6. Access the saved files in the training phrase
In the training phrase, MetaLDA saves four files in the ```<save_folder>```:
- top_words.txt: the top 50 words with the largest weights (phi) in each topic (the number of top words can be changed)
- train_alphabet.txt: the vocabulary of the training documents, the order of the words matches the index of phi.
- train_target_alphabet.txt: the vocabulary of the labels in the training documents, the order of the labels matches the index of lambda
- train_stats.mat: a [MAT-file](https://au.mathworks.com/help/matlab/matlab_env/save-load-and-delete-workspace-variables.html) of Matlab that saves the training statistics. [matfilerw](https://github.com/diffplug/matfilerw) for JAVA and [FileIO of Scipy](https://docs.scipy.org/doc/scipy/reference/tutorial/io.html) for Python are good tools to access MAT-files.  Note that Matlab is not required although it can directly load MAT-files. 
7. Inference on the testing documents
MetaLDA offers two kinds of inference:
- Ignore the words that exist in the testing documents but not in the training documents
		- ```java -cp ./target/metalda-0.1-jar-with-dependencies.jar hezhao.MetaLDAInfer --test-docs <testing_doc_mallet_location> --save-folder <save_folder> --compute-perplexity true```
		- ```<save_folder>```: same to the folder where the files are saved in the training phrase
		- ```--compute-perplexity```
				- true: MetaLDA will use one half of each testing document (every first words) to sample the document-topic distributions (theta) and the other half (every second words) to compute perplexity.
				- false: MetaLDA will use all the content of each testing document to sample the document-topic distributions (theta). No perplexity will be computed.
- Consider the words that exist in the testing documents but not in the training documents
		- ```java -cp ./target/metalda-0.1-jar-with-dependencies.jar hezhao.MetaLDAInferUnseen --test-docs <testing_doc_mallet_location> --save-folder <save_folder> --compute-perplexity true --word-features <binary_embedding_location>```
8. Access the saved files in the inference phrase
- If ```MetaLDAInfer``` is used, MetaLDA will save the testing statistics into 'test_stats.mat' in ```<save_folder>```
- If ```MetaLDAInferUnseen``` is used, MetaLDA will save the testing statistics into 'test_stats_unseen.mat' in ```<save_folder>```
