package topicmodels;

import cc.mallet.types.*;
import cc.mallet.util.CommandOption;
import cc.mallet.util.Randoms;
import com.google.common.base.Charsets;
import com.google.common.io.Files;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;

import java.io.*;
import java.util.*;


public class MetaLDATrain extends ParallelTopicModelHyper{





    protected int[][] docTables; //t

    protected int[][] wordTables; //t'


    protected double[][] lambda;

    protected double[][] delta;

    protected Randoms random;

    protected double mu_doc_normal = 1.0;

    protected double mu_doc_default = 1.0;

    protected int numDocFeatures;

    protected int docDefaultFeatureIndex;

    protected ArrayList<ArrayList<Integer>> featureDoc; //the docs where each label is active

    protected double[] docLogQ;


    protected double mu_word_normal = 1.0;

    protected double mu_word_default = 1.0;

    protected int numWordFeatures;


    protected int wordDefaultFeatureIndex;

    protected ArrayList<ArrayList<Integer>> featureWord; //the words where each feature is active

    protected double[] wordLogQ;

    protected ArrayList<int[]> wordFeatureList; // just for the words in the training docs


    /**
     * 1. full alpha matrix optimised with doc labels
     * 2. Asymmetric alpha (each column (topic) optimised with default doc labels)
     * 3. Symmetric alpha (same value for all alpha)
     * 0. fixed.
     * */

    protected int sampleAlphaMethod;

    /**
     * 1. full beta matrix optimised with word features
     * 2. Asymmetric beta (each row (topic) optimised with default word features)
     * 3. Symmetric beta (same value for all beta)
     * 0. fixed.
     * */
    protected int sampleBetaMethod;


    protected HashMap<String, int[]> wordFeatureMap; //for all the words
    
	
	public MetaLDATrain(int numberOfTopics) {
		super(numberOfTopics);
	}
	
	
	public MetaLDATrain(int numberOfTopics, double alphaSum, double beta) {
		super(numberOfTopics, alphaSum, beta);

    }


	@Override
	public void optimizeAlpha(WorkerRunnableHyper[] runnables) {



        if(this.sampleAlphaMethod == 1)
        {
            this.optimizeAlphaWithDocLabel();
        }
        else if(this.sampleAlphaMethod == 2)
        {
            this.optimizeAlphaAsymmetric();
        }
        else if(this.sampleAlphaMethod == 3)
        {
            this.optimizeAlphaSymmetric();
        }
        else if(this.sampleAlphaMethod == 0)
        {
            return;
        }
        else
        {
            logger.warning("wrong argument of sampleAlphaMethod!");
        }

	}


	public void optimizeAlphaWithDocLabel()
    {

        assert(this.sampleAlphaMethod == 1);
        for(int feature = 0; feature < numDocFeatures; feature ++)
            Arrays.fill(this.docTables[feature],0);

        int[] docTopicCount = new int[numTopics];

        for (int doc=0; doc < data.size(); doc++) {

		    /* get topic counts*/
            FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).instance.getData();
            LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
            FeatureVector feature = (FeatureVector) data.get(doc).instance.getTarget();

            int[] featureIndices = this.sampleAlphaMethod == 2? new int[0] : feature.getIndices();

            int[] oneDocTopics = topicSequence.getFeatures();


            int docLength = tokenSequence.getLength();


            if (docLength <= 0)
            {
                logger.warning("document length is zero at " + Integer.toString(doc));
            }

            else
            {

                int topic;

                int newTable;

                Arrays.fill(docTopicCount, 0);

                for (int position = 0; position < docLength; position++) {
                    if (oneDocTopics[position] == ParallelTopicModelHyper.UNASSIGNED_TOPIC) {
                        continue;
                    }
                    topic = oneDocTopics[position];

                    docTopicCount[topic]++;

                }

                /* sample tables */
                for (topic = 0; topic < numTopics; topic++)
                {
                    if (docTopicCount[topic] > 0)
                    {
                        newTable = 1;
                        for (int j = 1; j < docTopicCount[topic]; j++)
                        {
                            newTable += random.nextDouble() < (alpha[doc][topic] / (alpha[doc][topic] + j)) ? 1 : 0;
                        }
                        for (int f = 0; f < featureIndices.length; f++)
                            this.docTables[featureIndices[f]][topic] += newTable;
                        docTables[docDefaultFeatureIndex][topic] += newTable;
                    }

                }

                /*sample q*/

                this.docLogQ[doc] = - Math.log(random.nextBeta(this.alphaSum[doc], docLength ));
            
            }


        }


        double mu;

        double temp;

        double newLambda;

        /*sample lambda and update alpha*/
        for(int feature = 0; feature < numDocFeatures; feature ++ )
        {
            mu = (feature == docDefaultFeatureIndex) ? mu_doc_default:mu_doc_normal;
            for(int topic = 0; topic < this.numTopics; topic++)
            {
                newLambda = random.nextGamma(mu + this.docTables[feature][topic], 1);
                temp = 0;
                for(int doc : featureDoc.get(feature))
                {
                    temp += alpha[doc][topic] * docLogQ[doc];
                }

                newLambda = newLambda / (temp + 1.0 * this.lambda[feature][topic]);


                for(int doc : featureDoc.get(feature))
                {
                    alpha[doc][topic] *= newLambda;
                }

                this.lambda[feature][topic] *= newLambda;

                assert(!Double.isNaN(this.lambda[feature][topic]));


            }

        }

        Arrays.fill(this.alphaSum, 0);
        for(int doc = 0; doc < data.size(); doc ++)
        {
            for(int topic = 0; topic < this.numTopics; topic++)
            {
                this.alphaSum[doc] += this.alpha[doc][topic];
            }
        }
    }


    public void optimizeAlphaAsymmetric()
    {

        assert(this.sampleAlphaMethod == 2);
        for(int feature = 0; feature < numDocFeatures; feature ++)
            Arrays.fill(this.docTables[feature],0);

        int[] docTopicCount = new int[numTopics];

        for (int doc=0; doc < data.size(); doc++) {

		    /* get topic counts*/
            FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).instance.getData();
            LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
            FeatureVector feature = (FeatureVector) data.get(doc).instance.getTarget();

            int[] oneDocTopics = topicSequence.getFeatures();


            int docLength = tokenSequence.getLength();


            if (docLength <= 0)
            {
                logger.warning("document length is zero at " + Integer.toString(doc));
            }

            else
            {
                int topic;

                int newTable;

                Arrays.fill(docTopicCount, 0);

                for (int position = 0; position < docLength; position++) {
                    if (oneDocTopics[position] == ParallelTopicModelHyper.UNASSIGNED_TOPIC) {
                        continue;
                    }
                    topic = oneDocTopics[position];

                    docTopicCount[topic]++;

                }

                /* sample tables */
                for (topic = 0; topic < numTopics; topic++)
                {
                    if (docTopicCount[topic] > 0)
                    {
                        newTable = 1;
                        for (int j = 1; j < docTopicCount[topic]; j++)
                        {
                            newTable += random.nextDouble() < (alpha[doc][topic] / (alpha[doc][topic] + j)) ? 1 : 0;
                        }
                        docTables[docDefaultFeatureIndex][topic] += newTable;
                    }

                }

                /*sample q*/
                this.docLogQ[doc] = - Math.log(random.nextBeta(this.alphaSum[doc], docLength ));

            }
        }


        double mu;

        double temp;

        double newLambda;

        /*sample lambda and update alpha*/

        for(int topic = 0; topic < this.numTopics; topic++)
        {
            newLambda = random.nextGamma(mu_doc_default + this.docTables[docDefaultFeatureIndex][topic], 1);
            temp = 0;
            for(int doc : featureDoc.get(docDefaultFeatureIndex))
            {
                temp += alpha[doc][topic] * docLogQ[doc];
            }

            newLambda = newLambda / (temp + 1.0 * this.lambda[docDefaultFeatureIndex][topic]);


            for(int doc : featureDoc.get(docDefaultFeatureIndex))
            {
                alpha[doc][topic] *= newLambda;
            }

            this.lambda[docDefaultFeatureIndex][topic] *= newLambda;

            assert(!Double.isNaN(this.lambda[docDefaultFeatureIndex][topic]));

        }



        Arrays.fill(this.alphaSum, 0);
        for(int doc = 0; doc < data.size(); doc ++)
        {
            for(int topic = 0; topic < this.numTopics; topic++)
            {
                this.alphaSum[doc] += this.alpha[doc][topic];
            }
        }
    }

    public void optimizeAlphaSymmetric()
    {

        assert(this.sampleAlphaMethod == 3);
        for(int feature = 0; feature < numDocFeatures; feature ++)
            Arrays.fill(this.docTables[feature],0);

        int[] docTopicCount = new int[numTopics];


        int sumTable = 0;

        double sumDocLogQ = 0.0;

        for (int doc=0; doc < data.size(); doc++) {

		    /* get topic counts*/
            FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).instance.getData();
            LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
            FeatureVector feature = (FeatureVector) data.get(doc).instance.getTarget();

            int[] oneDocTopics = topicSequence.getFeatures();


            int docLength = tokenSequence.getLength();


            if (docLength <= 0)
            {
                logger.warning("document length is zero at " + Integer.toString(doc));
            }

            else

            {

                int topic;

                int newTable;

                Arrays.fill(docTopicCount, 0);

                for (int position = 0; position < docLength; position++) {
                    if (oneDocTopics[position] == ParallelTopicModelHyper.UNASSIGNED_TOPIC) {
                        continue;
                    }
                    topic = oneDocTopics[position];

                    docTopicCount[topic]++;

                }

                /* sample tables */
                for (topic = 0; topic < numTopics; topic++)
                {
                    if (docTopicCount[topic] > 0)
                    {
                        newTable = 1;
                        for (int j = 1; j < docTopicCount[topic]; j++)
                        {
                            newTable += random.nextDouble() < (alpha[doc][topic] / (alpha[doc][topic] + j)) ? 1 : 0;
                        }
                        sumTable += newTable;
                    }

                }

                /*sample q*/
                this.docLogQ[doc] = - Math.log(random.nextBeta(this.alphaSum[doc], docLength ));
                sumDocLogQ += this.docLogQ[doc];
            }
        }


        double oneAlpha = random.nextGamma(mu_doc_default + sumTable) / (1/mu_doc_default + sumDocLogQ * numTopics);

        Arrays.fill(this.alphaSum, 0);
        for(int doc = 0; doc < data.size(); doc ++)
        {
            for(int topic = 0; topic < this.numTopics; topic++)
            {
                this.alpha[doc][topic] = oneAlpha;
                this.alphaSum[doc] += this.alpha[doc][topic];
            }
        }
    }


	@Override
	public void optimizeBeta(WorkerRunnableHyper[] runnables) 
	{

        if(this.sampleBetaMethod == 1)
        {
            this.optimizeBetaWithWordFeature();
        }
        else if(this.sampleBetaMethod == 2)
        {
            this.optimizeBetaAsymmetric();
        }
        else if(this.sampleBetaMethod == 3)
        {
            this.optimizeBetaSymmetric();
        }
        else if(this.sampleBetaMethod == 0)
        {
            return;
        }
        else
        {
            logger.warning("wrong argument of sampleBetaMethod!");
        }




	}


	public void optimizeBetaWithWordFeature()
    {
        for(int feature = 0; feature < numWordFeatures; feature ++) {
            Arrays.fill(this.wordTables[feature], 0);
        }


        /*sample tables*/
        int newTable;

        for (int type = 0; type < numTypes; type ++)
        {
            int[] currentTypeTopicCounts = typeTopicCounts[type];
            for(int index = 0; index < currentTypeTopicCounts.length; index ++)
            {

                int currentTopic = currentTypeTopicCounts[index] & topicMask;
                int currentValue = currentTypeTopicCounts[index] >> topicBits;


                if (currentValue > 0)
                {
                    newTable = 1;

                    for (int j = 1; j < currentValue; j++) {
                        newTable += random.nextDouble() < (beta[currentTopic][type] / (beta[currentTopic][type] + j)) ? 1 : 0;
                    }



                    if (this.wordFeatureList.size() > 0) {

                        int[] featureIndices = this.wordFeatureList.get(type);
                        if (featureIndices != null) {
                            for (int f = 0; f < featureIndices.length; f++)
                                this.wordTables[featureIndices[f]][currentTopic] += newTable;

                        }
                    }
                    wordTables[wordDefaultFeatureIndex][currentTopic] += newTable;
                }




            }

        }

		/*sample q*/

        for (int topic = 0; topic < numTopics; topic ++)
        {
            if(this.tokensPerTopic[topic] > 0) // if the topic is active
                this.wordLogQ[topic] = -Math.log(random.nextBeta(this.betaSum[topic], this.tokensPerTopic[topic]));

        }


        /*sample delta and update beta*/

        double mu;

        double temp;

        double newDelta;


        for(int feature = 0; feature < numWordFeatures; feature ++ )
        {
            mu = (feature == wordDefaultFeatureIndex) ? mu_word_default:mu_word_normal;
            for(int topic = 0; topic < numTopics; topic++)
            {
                if (this.tokensPerTopic[topic] > 0) {
                    newDelta = random.nextGamma(mu + this.wordTables[feature][topic], 1);
                    temp = 0;
                    for (int type : this.featureWord.get(feature)) {
                        temp += beta[topic][type] * wordLogQ[topic];
                    }
                    newDelta = newDelta / (temp + 1.0 * this.delta[feature][topic]);
                    for (int type : featureWord.get(feature)) {
                        beta[topic][type] *= newDelta;

                    }

                    this.delta[feature][topic] *= newDelta;

                }






            }


        }

        Arrays.fill(this.betaSum, 0);


        for(int topic = 0; topic < this.numTopics; topic++)
        {
            for(int type = 0; type < numTypes; type ++)
            {
                this.betaSum[topic] += this.beta[topic][type];
            }
        }
    }


    public void optimizeBetaAsymmetric()
    {
        for(int feature = 0; feature < numWordFeatures; feature ++) {
            Arrays.fill(this.wordTables[feature], 0);
        }


        /*sample tables*/
        int newTable;

        for (int type = 0; type < numTypes; type ++)
        {
            int[] currentTypeTopicCounts = typeTopicCounts[type];
            for(int index = 0; index < currentTypeTopicCounts.length; index ++)
            {

                int currentTopic = currentTypeTopicCounts[index] & topicMask;
                int currentValue = currentTypeTopicCounts[index] >> topicBits;


                if (currentValue > 0)
                {
                    newTable = 1;

                    for (int j = 1; j < currentValue; j++) {
                        newTable += random.nextDouble() < (beta[currentTopic][type] / (beta[currentTopic][type] + j)) ? 1 : 0;
                    }

                    wordTables[wordDefaultFeatureIndex][currentTopic] += newTable;
                }




            }

        }

		/*sample q*/

        for (int topic = 0; topic < numTopics; topic ++)
        {
            if (this.tokensPerTopic[topic] > 0)
                this.wordLogQ[topic] = -Math.log(random.nextBeta(this.betaSum[topic], this.tokensPerTopic[topic]));

        }


        /*sample delta and update beta*/

        double mu;

        double temp;

        double newDelta;



        for(int topic = 0; topic < numTopics; topic++)
        {
            if (this.tokensPerTopic[topic] > 0) {
                newDelta = random.nextGamma(mu_word_default + this.wordTables[wordDefaultFeatureIndex][topic], 1);
                temp = 0;
                for (int type : this.featureWord.get(wordDefaultFeatureIndex)) {
                    temp += beta[topic][type] * wordLogQ[topic];
                }

                newDelta = newDelta / (temp + 1.0 * this.delta[wordDefaultFeatureIndex][topic]);


                for (int type : featureWord.get(wordDefaultFeatureIndex)) {
                    beta[topic][type] *= newDelta;

                }

                this.delta[wordDefaultFeatureIndex][topic] *= newDelta;
            }


        }




        Arrays.fill(this.betaSum, 0);


        for(int topic = 0; topic < this.numTopics; topic++)
        {
            for(int type = 0; type < numTypes; type ++)
            {
                this.betaSum[topic] += this.beta[topic][type];
            }
        }
    }



    public void optimizeBetaSymmetric()
    {
        for(int feature = 0; feature < numWordFeatures; feature ++) {
            Arrays.fill(this.wordTables[feature], 0);
        }


        /*sample tables*/
        int newTable;

        int sumTable = 0;

        for (int type = 0; type < numTypes; type ++)
        {
            int[] currentTypeTopicCounts = typeTopicCounts[type];
            for(int index = 0; index < currentTypeTopicCounts.length; index ++)
            {

                int currentTopic = currentTypeTopicCounts[index] & topicMask;
                int currentValue = currentTypeTopicCounts[index] >> topicBits;


                if (currentValue > 0)
                {
                    newTable = 1;

                    for (int j = 1; j < currentValue; j++) {
                        newTable += random.nextDouble() < (beta[currentTopic][type] / (beta[currentTopic][type] + j)) ? 1 : 0;
                    }

                    sumTable += newTable;
                }




            }

        }

		/*sample q*/

		double sumWordLogQ = 0.0;

        for (int topic = 0; topic < numTopics; topic ++)
        {

            if (this.tokensPerTopic[topic] > 0) {
                this.wordLogQ[topic] = -Math.log(random.nextBeta(this.betaSum[topic], this.tokensPerTopic[topic]));

                sumWordLogQ += this.wordLogQ[topic];
            }

        }




        double oneBeta = random.nextGamma(mu_word_default + sumTable) / (1/mu_word_default + sumWordLogQ * numTypes);


        Arrays.fill(this.betaSum, 0);


        for(int topic = 0; topic < this.numTopics; topic++)
        {
            for(int type = 0; type < numTypes; type ++)
            {
                this.beta[topic][type] = oneBeta;
                this.betaSum[topic] += this.beta[topic][type];
            }
        }
    }




    @Override
	public void addInstances(InstanceList training) {
		// TODO Auto-generated method stub
		super.addInstances(training);
		
		if (randomSeed == -1) {
			random = new Randoms((int)System.currentTimeMillis());
		}
		else {
			random = new Randoms(randomSeed);
		}


        numDocFeatures = this.sampleAlphaMethod == 2?1:data.get(0).instance.getTargetAlphabet().size() + 1;
        docDefaultFeatureIndex = numDocFeatures - 1;
		
		this.docTables = new int[numDocFeatures][];
		
		this.lambda = new double[numDocFeatures][];
		
        double mu;
        
        this.featureDoc = new ArrayList<ArrayList<Integer>>();
		for(int feature = 0; feature < numDocFeatures; feature ++)
		{
			this.lambda[feature] = new double[this.numTopics];
			this.docTables[feature] = new int[this.numTopics];
			
			mu = (feature == docDefaultFeatureIndex) ? mu_doc_default:mu_doc_normal;
			
            /*init lambda*/
			for (int topic = 0; topic < numTopics; topic ++)
			{
				this.lambda[feature][topic] = random.nextGamma(mu, 1);
			}
			this.featureDoc.add(new ArrayList<Integer>());
		}


		/*build index of docs where a label is active*/
        for (int doc=0; doc < data.size(); doc++) 
        {
            
			FeatureVector feature = (FeatureVector) data.get(doc).instance.getTarget();

			int[] featureIndices = this.sampleAlphaMethod == 2? new int[0] : feature.getIndices();
			for(int f = 0; f < featureIndices.length; f++)
			{
				this.featureDoc.get(featureIndices[f]).add(doc);
			}
			this.featureDoc.get(docDefaultFeatureIndex).add(doc);

        }
		
        this.docLogQ = new double[data.size()];

        /*init alpha*/
		if (this.sampleAlphaMethod == 1 || this.sampleAlphaMethod == 2)
        {

            for(int doc = 0; doc < data.size(); doc ++) {

                this.alphaSum[doc] = 0;
                FeatureVector feature = (FeatureVector) data.get(doc).instance.getTarget();
                int[] featureIndices = this.sampleAlphaMethod == 2? new int[0] : feature.getIndices();
                for (int topic = 0; topic < numTopics; topic++) {
                    alpha[doc][topic] = lambda[docDefaultFeatureIndex][topic];
                    for (int f = 0; f < featureIndices.length; f++) {
                        alpha[doc][topic] *= lambda[featureIndices[f]][topic];

                    }
                    this.alphaSum[doc] += this.alpha[doc][topic];
                }

            }

        }
        
	}

    public void addWordFeaturesFile(File wordFeatureFile)
    {

        List<String> lines;
        try {
            lines = Files.readLines(wordFeatureFile, Charsets.UTF_8);
        } catch (IOException e) {

            logger.warning("Read word feature file error!");
            lines = null;
        }


        /* a list of non-zero word features, for training phrase only, containing the words in training docs only*/
        this.wordFeatureList = new ArrayList<int[]>();

        int maxFeatureIndex = 0;

        /*parse input lines*/
        if (lines != null && this.sampleBetaMethod == 1) {

             /* a map of non-zero word features, for both training and testing, containing all words in the word feature file*/
             this.wordFeatureMap = new HashMap<String, int[]>();



            for (String line : lines) {
                String[] ls = line.split("\t");


                String[] fls = ls[1].trim().split(" ");

                int[] feature = new int[fls.length];

                for (int i = 0; i < fls.length; i++) {
                    feature[i] = Integer.parseInt(fls[i]);
                    if (feature[i] > maxFeatureIndex)
                        maxFeatureIndex = feature[i];
                }

                wordFeatureMap.put(ls[0], feature);


            }


            for (Iterator<Object> iter = this.alphabet.iterator(); iter.hasNext(); ) {
                String type = (String) iter.next();
                if (!wordFeatureMap.containsKey(type))
                {
                    this.wordFeatureList.add(null);
                }
                else
                {

                    this.wordFeatureList.add(wordFeatureMap.get(type));
                }
            }
            numWordFeatures = maxFeatureIndex + 1 + 1;


        }
        else
        {
            numWordFeatures = 1; //default feature
        }






        wordDefaultFeatureIndex = numWordFeatures - 1;

        this.wordTables = new int[numWordFeatures][];

        this.delta = new double[numWordFeatures][];

        double mu;

        this.featureWord = new ArrayList<ArrayList<Integer>>();

        for(int feature = 0; feature < numWordFeatures; feature ++)
        {
            this.featureWord.add(new ArrayList<Integer>());
            this.wordTables[feature] = new int[numTopics];
            this.delta[feature] = new double[numTopics];

            mu = (feature == wordDefaultFeatureIndex) ? mu_word_default:mu_word_normal;

            /*init delta*/
            for(int topic = 0; topic < numTopics; topic ++)
            {
                this.delta[feature][topic] = random.nextGamma(mu, 1);
            }
        }

        /*build index for words where a feature is active*/
        for (int type=0; type < this.numTypes; type++)
        {

            if (this.wordFeatureList.size() > 0) {
                int[] featureIndices = this.wordFeatureList.get(type);
                if (featureIndices != null) {
                    for (int f = 0; f < featureIndices.length; f++) {
                        this.featureWord.get(featureIndices[f]).add(type);
                    }
                }
            }
            this.featureWord.get(wordDefaultFeatureIndex).add(type);

        }

        this.wordLogQ = new double[numTopics];

        /*init beta*/
        if (this.sampleBetaMethod == 1 || this.sampleBetaMethod == 2) {
            for (int topic = 0; topic < numTopics; topic++) {
                this.betaSum[topic] = 0;

                for (int type = 0; type < numTypes; type++) {

                    this.beta[topic][type] = this.delta[wordDefaultFeatureIndex][topic];

                    if (this.wordFeatureList.size() > 0) {
                        int[] featureIndices = this.wordFeatureList.get(type);
                        if (featureIndices != null) {
                            for (int f = 0; f < featureIndices.length; f++) {
                                this.beta[topic][type] *= this.delta[featureIndices[f]][topic];
                            }
                        }
                    }
                    this.betaSum[topic] += this.beta[topic][type];
                }
            }
        }





    }



    public void setSampleAlphaMethod(int sampleAlphaMethod) {
		this.sampleAlphaMethod = sampleAlphaMethod;
	}




	public void setSampleBetaMethod(int sampleBetaMethod) {
		this.sampleBetaMethod = sampleBetaMethod;
		if (sampleBetaMethod == 1 || sampleBetaMethod == 2) //beta must be a matrix
		{
			this.isFullBeta = true;
		}
		else //beta can be a vector
		{
			this.isFullBeta = false;
		}
	}


    public void saveFiles(String folderName) throws IOException {


        File matSaveFile = new File(folderName + "/train_stats.mat");


        ArrayList saveList = new ArrayList();


        MLDouble alphaMat = null;

        alphaMat = new MLDouble("alpha", this.alpha);


        saveList.add( alphaMat );


        MLDouble betaMat = null;

        betaMat = new MLDouble("beta", this.beta);

        saveList.add(betaMat);

        MLDouble docTopicCounts = new MLDouble("doc_topic", this.getDocumentTopics(false, false));

        saveList.add(docTopicCounts);

        MLDouble topicTypeCounts = new MLDouble("topic_type", this.getTopicWords(false,false));

        saveList.add(topicTypeCounts);


        if (this.lambda != null)
        {
            MLDouble lambdaMat = new MLDouble("lambda", this.lambda);

            saveList.add(lambdaMat);


        }

        if (this.delta != null)
        {
            MLDouble deltaMat = new MLDouble("delta", this.delta);

            saveList.add(deltaMat);
        }



        saveList.add(new MLChar("sampleAlphaMethod", Integer.toString(this.sampleAlphaMethod)));
        saveList.add(new MLChar("sampleBetaMethod", Integer.toString(this.sampleBetaMethod)));

        saveList.add(new MLChar("numTopics", Integer.toString(this.numTopics)));

        saveList.add(new MLChar("numIterations", Integer.toString(this.numIterations)));

        saveList.add(new MLChar("burninPeriod", Integer.toString(this.burninPeriod)));

        saveList.add(new MLChar("optimizeAlphaInterval", Integer.toString(this.optimizeAlphaInterval)));
        saveList.add(new MLChar("optimizeBetaInterval", Integer.toString(this.optimizeBetaInterval)));





        new MatFileWriter(matSaveFile, saveList);





        File trainAlphabetFile = new File(folderName + "/train_alphabet.txt");

        PrintStream trainAlphabetOut = new PrintStream(trainAlphabetFile);

        trainAlphabetOut.println(data.get(0).instance.getDataAlphabet().toString());

        trainAlphabetOut.close();


        File trainTargetAlphabetFile = new File(folderName + "/train_target_alphabet.txt");

        PrintStream trainTargetAlphabetOut = new PrintStream(trainTargetAlphabetFile);

        trainTargetAlphabetOut.println(data.get(0).instance.getTargetAlphabet().toString());

        trainTargetAlphabetOut.close();



        this.printTopWords(new File(folderName + "/top_words.txt"),50, false);



    }



    static CommandOption.String docInputFile =
            new cc.mallet.util.CommandOption.String(MetaLDATrain.class, "train-docs", "FILENAME", true, null,
                    "The filename from which to read the list of training instances.  Use - for stdin.  " +
                            "The instances must be FeatureSequence, not FeatureVector", null);


    static CommandOption.String wordInputFile =
            new cc.mallet.util.CommandOption.String(MetaLDATrain.class, "word-features", "FILENAME", true, null,
                    "The filename from which to read the word features.  Use - for stdin.", null);

    static CommandOption.String saveFolderName =
            new cc.mallet.util.CommandOption.String(MetaLDATrain.class, "save-folder", "FILENAME", true, null,
                    "the folder that saves the statistics", null);


    // Model parameters

    static CommandOption.Integer numIterationsOption =
            new CommandOption.Integer(MetaLDATrain.class, "num-iterations", "INTEGER", true, 2000,
                    "The number of iterations of Gibbs sampling. Default is 2000.", null);



    static CommandOption.Integer randomSeedOption =
            new CommandOption.Integer(MetaLDATrain.class, "random-seed", "INTEGER", true, -1,
                    "The random seed for the Gibbs sampler.  Default is -1, which will use the clock.", null);

    // Hyperparameters



    static CommandOption.Integer sampleAlphaMethodOption =
            new CommandOption.Integer(MetaLDATrain.class, "sample-alpha-method", "INTEGER", true, 0,
                    "0. fixed with initial value; 1. sample with doc labels; 2. sample with default label (asymmetric); 3. symmetric. Default is 0", null);

    static CommandOption.Integer sampleBetaMethodOption =
            new CommandOption.Integer(MetaLDATrain.class, "sample-beta-method", "INTEGER", true, 0,
                    "0. fixed with initial value; 1. sample with word features; 2. sample with default feature (asymmetric); 3. symmetric. Default is 0", null);

    static CommandOption.Integer sampleBetaIntervalOption =
            new CommandOption.Integer(MetaLDATrain.class, "sample-beta-interval", "INTEGER", true, 1,
                    "The number of iterations between sampling beta. Default is 1", null);

    static CommandOption.Integer sampleAlphaIntervalOption =
            new CommandOption.Integer(MetaLDATrain.class, "sample-alpha-interval", "INTEGER", true, 1,
                    "The number of iterations between sampling alpha. Default is 1", null);

    static CommandOption.Integer burninPeriodOption =
            new CommandOption.Integer(MetaLDATrain.class, "burn-in-period", "INTEGER", true, 10,
                    "The number of iterations in burn-in period before sampling alpha and/or beta. Default is 10", null);




    static CommandOption.Integer numTopicsOption =
            new CommandOption.Integer(MetaLDATrain.class, "num-topics", "INTEGER", true, 50,
                    "The number of topics. Default is 50", null);


    static CommandOption.Integer numThreadsOption =
            new CommandOption.Integer(MetaLDATrain.class, "num-threads", "INTEGER", true, 1,
                    "The number of threads. Default is 1", null);

    static CommandOption.Double initAlpha =
            new CommandOption.Double(MetaLDATrain.class, "initial-alpha", "DOUBLE", true, 0.1,
                    "The initial value of alpha. Default is 0.1", null);

    static CommandOption.Double initBeta =
            new CommandOption.Double(MetaLDATrain.class, "initial-beta", "DOUBLE", true, 0.01,
                    "The initial value of beta. Default is 0.01", null);




	public static void main(String[] args){


        CommandOption.setSummary (MetaLDATrain.class,
                "MetaLDA, a topic model that incorporates meta information");
        CommandOption.process(MetaLDATrain.class, args);


        if(docInputFile.value == null)
        {
            logger.warning("missing training documents!");
            CommandOption.getList(MetaLDATrain.class).printUsage(false);
            System.exit(-1);
        }

        if(sampleBetaMethodOption.value == 1 && wordInputFile.value == null)
        {
            logger.warning("missing word feature file!");
            CommandOption.getList(MetaLDATrain.class).printUsage(false);
            System.exit(-1);
        }

        if(saveFolderName.value == null)
        {
            logger.warning("missing saving folder!");
            CommandOption.getList(MetaLDATrain.class).printUsage(false);
            System.exit(-1);
        }


        MetaLDATrain metaLda = null;

        InstanceList training = InstanceList.load(new File(docInputFile.value));
        metaLda = new MetaLDATrain(numTopicsOption.value, initAlpha.value * numTopicsOption.value, initBeta.value);

        metaLda.setSampleAlphaMethod(sampleAlphaMethodOption.value);

        metaLda.setSampleBetaMethod(sampleBetaMethodOption.value);

        metaLda.setOptimizeInterval(sampleAlphaIntervalOption.value, sampleBetaIntervalOption.value);

        metaLda.setBurninPeriod(burninPeriodOption.value);

        metaLda.setNumIterations(numIterationsOption.value);

        metaLda.setNumThreads(numThreadsOption.value);

        metaLda.setRandomSeed(randomSeedOption.value);


        metaLda.addInstances(training);

        //do not display topics while training
        metaLda.setTopicDisplay(0,0);


        if(wordInputFile.value != null)
        {
            metaLda.addWordFeaturesFile(new File(wordInputFile.value));
        }

        try {
            metaLda.estimate();
        } catch (IOException e) {
            e.printStackTrace();
        }

        if(saveFolderName.value != null && metaLda != null)
        {


            File saveDir = new File(saveFolderName.value);

            if (!saveDir.exists()) {

                saveDir.mkdir();

            }

            try {
                metaLda.saveFiles(saveFolderName.value);

            } catch (IOException e) {
                e.printStackTrace();
            }

 
        }













    }

}
