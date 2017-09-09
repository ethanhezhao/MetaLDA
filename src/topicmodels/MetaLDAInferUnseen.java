package topicmodels;

import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.*;
import cc.mallet.util.CommandOption;
import cc.mallet.util.Randoms;
import com.google.common.base.Charsets;
import com.google.common.io.Files;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;

import java.io.*;
import java.util.*;

public class MetaLDAInferUnseen implements Serializable {

	protected int numTopics;



	protected int numTypes;

	protected double[][] lambda;

	protected double[] oneAlpha;

	protected double[][] alpha;

	protected double[][] phi;


	Alphabet trainAlphabet;

	Alphabet trainTargetAlphabet;

	protected Randoms random = null;


	int sampleAlphaMethod;

	int numDocFeatures;

	int docDefaultFeatureIndex;

    protected double[][] delta;

    protected int numTrainTypes;


    int sampleBetaMethod;


    int numWordFeatures;


    int wordDefaultFeatureIndex;

    double[][] trainTopicType;

    double[] trainTopicTypeSum;

    double[][] trainBeta;

    double[][] testPhi;

    double[][] testBeta;


    /**
     * sampleAlphaMethod = 0, 2, 3, 4, 5
     * sampleBetaMethod = 0, 2, 3, 4
     * */
    public MetaLDAInferUnseen(Alphabet alphabet, Alphabet targetAlphabet,
                                           double[] alpha, double[][] topicType, double[][] beta, int sampleAlphaMethod,
                              int sampleBetaMethod) {



        this.trainAlphabet = alphabet;

        this.trainTargetAlphabet = targetAlphabet;

        numTopics = topicType.length;
        numTrainTypes =  topicType[0].length;



        this.oneAlpha = alpha;


        random = new Randoms();

        this.trainTopicType = topicType;

        this.trainBeta = beta;

        this.sampleAlphaMethod = sampleAlphaMethod;
        this.sampleBetaMethod = sampleBetaMethod;

        this.trainTopicTypeSum = new double[numTopics];

        for (int topic = 0; topic < numTopics; topic++)
        {
            for(int trainType = 0; trainType < numTrainTypes; trainType ++)
            {
                this.trainTopicTypeSum[topic] += this.trainTopicType[topic][trainType];
            }
        }


    }

    /**
     * sampleAlphaMethod = 1
     * sampleBetaMethod = 0, 2, 3, 4
     * */
    public MetaLDAInferUnseen(Alphabet alphabet, Alphabet targetAlphabet,
                                           double[][] lambda, double[][] topicType, double[][] beta,
                              int sampleAlphaMethod, int sampleBetaMethod) {

        this(alphabet, targetAlphabet, new double[0], topicType, beta, sampleAlphaMethod, sampleBetaMethod);

        this.lambda = lambda;


        this.numDocFeatures = lambda.length;

        this.docDefaultFeatureIndex = this.numDocFeatures - 1;

    }

    /**
     * sampleAlphaMethod = 0, 2, 3, 4, 5
     * sampleBetaMethod = 1
     * */
    public MetaLDAInferUnseen(Alphabet alphabet, Alphabet targetAlphabet,
                                           double[] alpha, double[][] topicType, double[][] delta, double[][] trainBeta,
                              int sampleAlphaMethod, int sampleBetaMethod)
    {
        this(alphabet, targetAlphabet, alpha, topicType, trainBeta, sampleAlphaMethod, sampleBetaMethod);
        this.delta = delta;

        this.numWordFeatures = delta.length;

        this.wordDefaultFeatureIndex = this.numWordFeatures - 1;


    }


    /**
     * sampleAlphaMethod = 1
     * sampleBetaMethod = 1
     * */
    public MetaLDAInferUnseen(Alphabet alphabet, Alphabet targetAlphabet,
                                           double[][] lambda, double[][] topicType, double[][] delta,
                              double[][] trainBeta, int sampleAlphaMethod, int sampleBetaMethod) {

        this(alphabet, targetAlphabet, lambda, topicType, trainBeta, sampleAlphaMethod, sampleBetaMethod);



        this.delta = delta;


        this.numWordFeatures = delta.length;

        this.wordDefaultFeatureIndex = this.numWordFeatures - 1;

    }








    public void setRandomSeed(int seed) {
        if (seed == -1)
            random = new Randoms();
        else
            random = new Randoms(seed);
    }

    /** Inference on a test doc*/
    public double[] getSampledDistribution(Instance testInstance, Alphabet testAlphabet, FeatureSequence topicSequence, double[] alpha, int numIterations,
                                           int thinning, int burnIn, boolean isOnEveryFirstWord) {


        FeatureSequence tokenSequence = (FeatureSequence) testInstance.getData();



        int[] oneDocTopics = topicSequence.getFeatures();


        int docLength = tokenSequence.size();

        int[] localTopicCounts = new int[numTopics];


        for (int position = 0; position < docLength; position++) {
            if (isOnEveryFirstWord && position % 2 == 1)
                continue;
            localTopicCounts[oneDocTopics[position]]++;
        }




        double score, sum = 0.0;

        double[] topicTermScores = new double[numTopics];

        int oldTopic, newTopic;

        double[] result = new double[numTopics];

        int avgCount = 0;

        for (int iteration = 1; iteration <= numIterations; iteration++) {

            //  Iterate over the positions (words) in the document
            for (int position = 0; position < docLength; position++) {

                if (isOnEveryFirstWord && position % 2 == 1)
                    continue;


                int testType = tokenSequence.getIndexAtPosition(position);


                oldTopic = oneDocTopics[position];


                localTopicCounts[oldTopic]--;

                // Now calculate and add up the scores for each topic for this word
                sum = 0.0;

                // Here's where the math happens! Note that overall performance is
                //  dominated by what you do in this loop.
                for (int topic = 0; topic < numTopics; topic++) {
                    score =
                            (alpha[topic] + localTopicCounts[topic]) * this.testPhi[topic][testType];

                    sum += score;
                    topicTermScores[topic] = score;
                }

                // Choose a random point between 0 and the sum of all topic scores
                double sample = random.nextUniform() * sum;

                // Figure out which topic contains that point
                newTopic = -1;
                while (sample > 0.0) {
                    newTopic++;
                    sample -= topicTermScores[newTopic];
                }

                // Make sure we actually sampled a topic
                if (newTopic == -1) {
                    throw new IllegalStateException ("Topic Inferencer: New topic not sampled.");
                }


                // Put that new topic into the counts
                oneDocTopics[position] = newTopic;
                localTopicCounts[newTopic]++;
            }

            if (iteration > burnIn &&
                    (iteration - burnIn) % thinning == 0) {
                avgCount ++;
                // Save a sample
                for (int topic=0; topic < numTopics; topic++) {
                    result[topic] += alpha[topic] + localTopicCounts[topic];
                }
            }
        }


        double sumResult = 0;
        for (int topic=0; topic < numTopics; topic++) {
            result[topic] /= avgCount;

            sumResult += result[topic];

        }
        for (int topic=0; topic < numTopics; topic++) {
            result[topic] /= sumResult;
        }



        return result;
    }


    /** Get the inferred topic distributions of the test docs*/
    public double[][] getInferredDistributions(InstanceList testInstances,  HashMap<String, int[]> wordFeatureVoc,
                                               int numIterations, int thinning, int burnIn, boolean isOnEveryFirstWord
    ) {

        ArrayList<TopicAssignment> data = new ArrayList<TopicAssignment>();

        LabelAlphabet topicAlphabet = newLabelAlphabet (numTopics);


        Alphabet testAlphabet = testInstances.getAlphabet();

		/*randomly init topics for the test docs*/

        int testNumTypes = testAlphabet.size();

        for (Instance instance : testInstances) {

            FeatureSequence tokens = (FeatureSequence) instance.getData();
            LabelSequence topicSequence =
                    new LabelSequence(topicAlphabet, new int[ tokens.size() ]);

            int[] topics = topicSequence.getFeatures();
            for (int position = 0; position < tokens.size(); position++) {

                if (isOnEveryFirstWord && position % 2 == 1)
                    continue;
                int topic = random.nextInt(numTopics);
                topics[position] = topic;
            }

            TopicAssignment t = new TopicAssignment (instance, topicSequence);
            data.add (t);
        }

		/*compute beta and phi for the test docs*/
        this.testPhi = new double[numTopics][];

        this.testBeta = new double[numTopics][];

        for(int topic = 0; topic < numTopics; topic++)
        {
            this.testPhi[topic] = new double[testNumTypes];
            this.testBeta[topic] = new double[testNumTypes];
        }


        if (this.sampleBetaMethod == 1 && wordFeatureVoc != null) {




			/* compute beta according to the features of the words in the test docs*/
            for(int testType = 0; testType < testNumTypes; testType ++)
            {

                String testWord = (String)testAlphabet.lookupObject(testType);

                if(wordFeatureVoc.containsKey(testWord))
                {
                    for (int topic = 0; topic < numTopics; topic++)
                    {

                        this.testBeta[topic][testType] = this.delta[wordDefaultFeatureIndex][topic];


                        int[] featureIndices = this.sampleBetaMethod == 2? null : wordFeatureVoc.get(testWord);;
                        if (featureIndices != null) {
                            for (int f = 0; f < featureIndices.length; f++) {
                                this.testBeta[topic][testType] *= this.delta[featureIndices[f]][topic];
                            }
                        }


                    }

                }


            }

        }
        else if(this.sampleBetaMethod == 0 || this.sampleBetaMethod == 3 || this.sampleBetaMethod == 4)
        {
            for(int topic = 0; topic < numTopics; topic++)
            {

                Arrays.fill(this.testBeta[topic],this.trainBeta[0][0]);

            }
        }
        else if(this.sampleBetaMethod == 2)
        {
            for(int topic = 0; topic < numTopics; topic++)
            {

                Arrays.fill(this.testBeta[topic],this.trainBeta[topic][0]);

            }
        }
        else
        {
            //error
        }

        double[] testPhiSum = new double[numTopics];

        for (int topic = 0; topic < numTopics; topic++)
        {
            for(int testType = 0; testType < testNumTypes; testType ++)
            {
                String testWord = (String)testAlphabet.lookupObject(testType);

                int trainType = this.trainAlphabet.lookupIndex(testWord,false);

                if (trainType == -1 || trainType >= numTrainTypes) //unseen words
                {

                    this.testPhi[topic][testType] = this.testBeta[topic][testType]; // no counts
                }
                else //seen words
                {
                    this.testPhi[topic][testType] = ( this.trainTopicType[topic][trainType] + this.trainBeta[topic][trainType]); // with counts
                }
                testPhiSum[topic] += this.testPhi[topic][testType];

            }
        }

		/*normalise phi*/
        for (int topic = 0; topic < numTopics; topic++) {
            for (int testType = 0; testType < testNumTypes; testType++) {
                this.testPhi[topic][testType] /= testPhiSum[topic];
            }
        }



        double[][] docTopic = new double[testInstances.size()][];



        int doc = 0;

        alpha = new double[testInstances.size()][];

        for (Instance instance: testInstances) {


            alpha[doc] = new double[numTopics];

            if(this.sampleAlphaMethod == 1)
            {
				/*compute alpha according to the labels of a test doc*/

                FeatureVector feature = (FeatureVector) instance.getTarget();
                int[] featureIndices = feature.getIndices();
                for (int topic = 0; topic < numTopics; topic ++)
                {
                    alpha[doc][topic] = 1.0;
                    for(int f = 0; f < featureIndices.length; f++)
                    {
                        String label = (String) instance.getTargetAlphabet().lookupObject(featureIndices[f]);

                        int fid = this.trainTargetAlphabet.lookupIndex(label,false);
                        if (fid < this.trainTargetAlphabet.size() && fid != -1)
                            alpha[doc][topic] *= lambda[fid][topic];
                    }
                    alpha[doc][topic] *= lambda[docDefaultFeatureIndex][topic];
                }


            }
            else
            {
                alpha[doc] = this.oneAlpha;
            }

            docTopic[doc] =
                    this.getSampledDistribution(instance, testAlphabet, data.get(doc).topicSequence, alpha[doc], numIterations,
                            thinning, burnIn, isOnEveryFirstWord);



            doc++;
        }

        return docTopic;
    }

	public double computePerplexityForEverySecondWord(InstanceList instances, Alphabet testAlphabet, double[][] testTheta, double[][] phi)
    {

        double perplexity = 0.0;

        int i = 0;

        int totalNumWords = 0;

        for (Instance instance: instances) {

            FeatureSequence tokenSequence = (FeatureSequence) instance.getData();




            int docLength = tokenSequence.size();

            int type;

            for (int position = 0; position < docLength; position++) {


                if (position % 2 == 0)
                    continue;

                type = tokenSequence.getIndexAtPosition(position);

                String word = (String) instance.getAlphabet().lookupObject(type);

                type = testAlphabet.lookupIndex(word,false);


                double prob = 0.0;
                for(int k = 0; k < numTopics; k++)
                {
                    prob += testTheta[i][k] * phi[k][type];
                }

                perplexity += Math.log(prob);

                totalNumWords ++;


            }
            i ++;

        }
        perplexity = Math.exp(-perplexity / totalNumWords);
        return perplexity;
    }


	// Serialization
	private static final long serialVersionUID = 1;



	private static LabelAlphabet newLabelAlphabet (int numTopics) {
		LabelAlphabet ret = new LabelAlphabet();
		for (int i = 0; i < numTopics; i++)
			ret.lookupIndex("topic"+i);
		return ret;
	}


    public HashMap<String,int[]> addWordFeaturesFile(File wordFeatureFile)
    {

        HashMap<String,int[]> wordFeatureVoc = null;
        List<String> lines;
        try {
            lines = Files.readLines(wordFeatureFile, Charsets.UTF_8);
        } catch (IOException e) {

            lines = null;
        }


        int maxFeatureIndex = 0;

        if (lines != null && this.sampleBetaMethod == 1) { // don't add word features if sampleBetaMethod == 2.

            wordFeatureVoc = new HashMap<String, int[]>();



            for (String line : lines) {
                String[] ls = line.split("\t");


                String[] fls = ls[1].trim().split(" ");

                int[] feature = new int[fls.length];

                for (int i = 0; i < fls.length; i++) {
                    feature[i] = Integer.parseInt(fls[i]);
                    if (feature[i] > maxFeatureIndex)
                        maxFeatureIndex = feature[i];
                }

                wordFeatureVoc.put(ls[0], feature);


            }




        }

        return wordFeatureVoc;






    }





    static CommandOption.String testDocInputFile =
            new CommandOption.String(MetaLDAInferUnseen.class, "test-docs", "FILENAME", true, null,
                    "The filename from which to read the list of the first part of testing instances.  Use - for stdin.  " +
                            "The instances must be FeatureSequence, not FeatureVector", null);


    static CommandOption.String saveFolderName =
            new CommandOption.String(MetaLDAInferUnseen.class, "save-folder", "FILENAME", true, null,
                    "the folder that saves the statistics", null);


    // Model parameters




    static CommandOption.Integer numInferenceIterationsOption =
            new CommandOption.Integer(MetaLDAInferUnseen.class, "num-inference-iterations", "INTEGER", true, 100,
                    "The number of inference iterations of Gibbs sampling. Default is 100.", null);


    static CommandOption.Integer randomSeedOption =
            new CommandOption.Integer(MetaLDAInferUnseen.class, "random-seed", "INTEGER", true, -1,
                    "The random seed for the Gibbs sampler.  Default is -1, which will use the clock.", null);



    static CommandOption.Boolean isComputePerplexity =
            new CommandOption.Boolean(MetaLDAInferUnseen.class, "compute-perplexity", "true|false", false, false,
                    "Whether infer the model on the every first words and compute perplexity on the every second words. Default is false.", null);

    static CommandOption.String wordInputFile =
            new cc.mallet.util.CommandOption.String(MetaLDAInferUnseen.class, "word-features", "FILENAME", true, null,
                    "The filename from which to read the word features.  Use - for stdin.  ", null);

    public static void main(String[] args){




        CommandOption.setSummary (MetaLDAInferUnseen.class,
                "Inference on new documents for MetaLDA");
        CommandOption.process (MetaLDAInferUnseen.class, args);





        if(testDocInputFile.value != null)
        {

            File testSaveFile = null;
            InstanceList testing = InstanceList.load(new File(testDocInputFile.value));

            MatFileReader matReader = null;
            try {
                matReader = new MatFileReader(saveFolderName.value + "/train_stats.mat");
            } catch (IOException e) {
                e.printStackTrace();
                System.err.println("Training info is not found in the folder!");
                System.exit(-1);
            }





            MLDouble testPhiMat = null;

            int sampleAlphaMethod = Integer.parseInt(((MLChar)matReader.getMLArray("sampleAlphaMethod")).getString(0));

            int sampleBetaMethod = Integer.parseInt(((MLChar)matReader.getMLArray("sampleBetaMethod")).getString(0));

            double[][] alpha = ((MLDouble) matReader.getMLArray("alpha")).getArray();


            double[][] beta = ((MLDouble) matReader.getMLArray("beta")).getArray();

            double[][] topicTypeCounts = ((MLDouble) matReader.getMLArray("topic_type")).getArray();

            double[][] phi = new double[beta.length][];

            for(int k = 0; k < phi.length; k++)
            {
                phi[k] = new double[beta[0].length];

                double sumPhi = 0;
                for (int v = 0; v < phi[k].length; v++)
                {
                    phi[k][v] = beta[k][v] + topicTypeCounts[k][v];
                    sumPhi += phi[k][v];
                }
                for (int v = 0; v < phi[k].length; v++)
                {
                    phi[k][v] /= sumPhi;

                }

            }

            double[][] lambda = null;
            if (sampleAlphaMethod == 1)
                lambda = ((MLDouble) matReader.getMLArray("lambda")).getArray();


            double[][] delta = null;
            if (sampleBetaMethod == 1)
                delta = ((MLDouble) matReader.getMLArray("delta")).getArray();




            Alphabet trainAlphabet = null;

            Alphabet trainTargetAlphabet = null;

            try {
                trainAlphabet = AlphabetFactory.loadFromFile(new File(saveFolderName.value + "/train_alphabet.txt"));
                trainTargetAlphabet = AlphabetFactory.loadFromFile(new File(saveFolderName.value + "/train_target_alphabet.txt"));
            } catch (IOException e) {
                e.printStackTrace();
            }



            MetaLDAInferUnseen inferencer = null;

            if (sampleAlphaMethod == 1)
            {

                if (sampleBetaMethod == 1)
                {
                    inferencer =  new MetaLDAInferUnseen(
                            trainAlphabet, trainTargetAlphabet,
                            lambda, topicTypeCounts, delta, beta, sampleAlphaMethod, sampleBetaMethod);
                }
                else
                {
                    inferencer =
                            new MetaLDAInferUnseen(trainAlphabet, trainTargetAlphabet, lambda, topicTypeCounts, beta, sampleAlphaMethod, sampleBetaMethod);
                }


            }
            else
            {
                if (sampleBetaMethod == 1)
                {
                    inferencer =  new MetaLDAInferUnseen(trainAlphabet,
                            trainTargetAlphabet, alpha[0], topicTypeCounts, delta, beta, sampleAlphaMethod, sampleBetaMethod);
                }
                else
                {
                    inferencer =  new MetaLDAInferUnseen(
                            trainAlphabet, trainTargetAlphabet,
                            alpha[0], topicTypeCounts, beta, sampleAlphaMethod, sampleBetaMethod);
                }
            }

            inferencer.setRandomSeed(randomSeedOption.value);

            HashMap<String,int[]> wordFeatureVoc = inferencer.addWordFeaturesFile(new File(wordInputFile.value));


            testSaveFile = new File(saveFolderName.value + "/test_stats_unseen.mat");


            MLDouble testThetaMat = null;

            double perplexity = 0.0;


            if (isComputePerplexity.value == true)
            {
                testThetaMat = new MLDouble("test_theta", inferencer.getInferredDistributions(testing, wordFeatureVoc, numInferenceIterationsOption.value, 10, 10, true));
                perplexity = inferencer.computePerplexityForEverySecondWord(testing, testing.getAlphabet(), testThetaMat.getArray(), inferencer.testPhi);
            }
            else
            {
                testThetaMat = new MLDouble("test_theta", inferencer.getInferredDistributions(testing, wordFeatureVoc, numInferenceIterationsOption.value, 10, 10, false));
            }


            ArrayList saveList = new ArrayList();

            saveList.add(testThetaMat);

            if (sampleAlphaMethod == 1) {
                MLDouble testAlphaMat = new MLDouble("test_alpha", inferencer.alpha);

                saveList.add(testAlphaMat);
            }

            if (sampleBetaMethod == 1)
            {
                MLDouble testBetaMat = new MLDouble("test_beta", inferencer.testBeta);

                saveList.add(testBetaMat);
            }

            if (perplexity > 0)
                saveList.add(new MLDouble("perplexity", new double[][]{new double[]{perplexity}}));

            try {
                new MatFileWriter(testSaveFile, saveList);
            } catch (IOException e) {
                e.printStackTrace();
            }





//            File testVocFile = new File(saveFolderName.value + "/test_alphabet.txt");
//
//            PrintStream testVocOut = null;
//            try {
//                testVocOut = new PrintStream(testVocFile);
//            } catch (FileNotFoundException e) {
//                e.printStackTrace();
//            }
//
//            testVocOut.println(testing.getAlphabet().toString());
//
//            testVocOut.close();
//
//
//            File testLabelFile = new File(saveFolderName.value + "/test_target_alphabet.txt");
//
//            PrintStream testLabelOut = null;
//            try {
//                testLabelOut = new PrintStream(testLabelFile);
//            } catch (FileNotFoundException e) {
//                e.printStackTrace();
//            }
//
//            testLabelOut.println(testing.getTargetAlphabet().toString());
//
//            testLabelOut.close();
        }








    }



}