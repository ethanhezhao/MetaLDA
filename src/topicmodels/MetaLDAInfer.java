package topicmodels;

import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.*;
import cc.mallet.util.CommandOption;
import cc.mallet.util.Randoms;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;

import java.io.*;
import java.util.ArrayList;

public class MetaLDAInfer implements Serializable {

	protected int numTopics;



	protected int numTrainTypes;

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


	/**
     * Inference for test docs, ignoring unseen words
     * */


	public MetaLDAInfer(Alphabet alphabet, Alphabet targetAlphabet,
                        double[] alpha, double[][] phi) {



		this.trainAlphabet = alphabet;

		this.trainTargetAlphabet = targetAlphabet;

		numTopics = phi.length;
		numTrainTypes =  phi[0].length;



		this.oneAlpha = alpha;


		random = new Randoms();

		this.phi = phi;

		this.sampleAlphaMethod = 0;
	}

	public MetaLDAInfer(Alphabet alphabet, Alphabet targetAlphabet,
                        double[][] lambda, double[][] phi) {

		this(alphabet, targetAlphabet, new double[0], phi);

		this.lambda = lambda;

		this.sampleAlphaMethod = 1;

		this.numDocFeatures = lambda.length;

		this.docDefaultFeatureIndex = this.numDocFeatures - 1;

	}


	public void setRandomSeed(int seed) {
	    if (seed == -1)
            random = new Randoms();
        else
		    random = new Randoms(seed);
	}


	/** Inference on a test doc*/
	public double[] getSampledDistribution(Instance instance, FeatureSequence topicSequence, double[] alpha, int numIterations,
										   int thinning, int burnIn, boolean isOnEveryFirstWord) {


		FeatureSequence tokenSequence = (FeatureSequence) instance.getData();



		int[] oneDocTopics = topicSequence.getFeatures();


		int docLength = tokenSequence.size();

		int[] localTopicCounts = new int[numTopics];

		int type;

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

				type = tokenSequence.getIndexAtPosition(position);

				String word = (String) instance.getAlphabet().lookupObject(type);

				type = this.trainAlphabet.lookupIndex(word,false);

				if (type == -1 || type >= numTrainTypes) {continue;}

				oldTopic = oneDocTopics[position];


				localTopicCounts[oldTopic]--;

				// Now calculate and add up the scores for each topic for this word
				sum = 0.0;

				// Here's where the math happens! Note that overall performance is
				//  dominated by what you do in this loop.
				for (int topic = 0; topic < numTopics; topic++) {
					score =
							(alpha[topic] + localTopicCounts[topic]) * phi[topic][type];

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

	public double[][] getInferredDistributions(InstanceList instances,
										   int numIterations, int thinning, int burnIn,
										   boolean isOnEveryFirstWord) {

		ArrayList<TopicAssignment> data = new ArrayList<TopicAssignment>();

		LabelAlphabet topicAlphabet = newLabelAlphabet (numTopics);

		int type;

		/*randomly init topics for the test docs*/
		for (Instance instance : instances) {

			FeatureSequence tokens = (FeatureSequence) instance.getData();
			LabelSequence topicSequence =
					new LabelSequence(topicAlphabet, new int[ tokens.size() ]);

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < tokens.size(); position++) {


				type = tokens.getIndexAtPosition(position);

				String word = (String) instance.getAlphabet().lookupObject(type);

				type = this.trainAlphabet.lookupIndex(word,false);

				if (type == -1 || type >= numTrainTypes) {continue;}

				int topic = random.nextInt(numTopics);
				topics[position] = topic;
			}

			TopicAssignment t = new TopicAssignment (instance, topicSequence);
			data.add (t);
		}


		double[][] docTopic = new double[instances.size()][];



		int doc = 0;

		alpha = new double[instances.size()][];

		for (Instance instance: instances) {



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
					getSampledDistribution(instance, data.get(doc).topicSequence, alpha[doc], numIterations,
							thinning, burnIn, isOnEveryFirstWord);



			doc++;
		}

		return docTopic;
	}

	public double computePerplexityForEverySecondWord(InstanceList instances, double[][] testTheta, double[][] phi)
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

                type = this.trainAlphabet.lookupIndex(word,false);

                if (type == -1 || type >= numTrainTypes) {
                    continue;
                }

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







    static CommandOption.String testDocInputFile =
            new cc.mallet.util.CommandOption.String(MetaLDAInfer.class, "test-docs", "FILENAME", true, null,
                    "The filename from which to read the list of testing instances.  Use - for stdin.  " +
                            "The instances must be FeatureSequence, not FeatureVector", null);


    static CommandOption.String saveFolderName =
            new cc.mallet.util.CommandOption.String(MetaLDAInfer.class, "save-folder", "FILENAME", true, null,
                    "the folder that saves the statistics", null);


    // Model parameters




    static CommandOption.Integer numInferenceIterationsOption =
            new CommandOption.Integer(MetaLDAInfer.class, "num-iterations", "INTEGER", true, 200,
                    "The number of inference iterations of Gibbs sampling. Default is 200.", null);


    static CommandOption.Integer randomSeedOption =
            new CommandOption.Integer(MetaLDAInfer.class, "random-seed", "INTEGER", true, -1,
                    "The random seed for the Gibbs sampler.  Default is -1, which will use the clock.", null);



    static CommandOption.Boolean isComputePerplexity =
            new CommandOption.Boolean(MetaLDAInfer.class, "compute-perplexity", "true|false", false, false,
                    "Whether infer the model on the every first words and compute perplexity on the every second words. Default is false.", null);

    public static void main(String[] args){




        CommandOption.setSummary (MetaLDAInfer.class,
                "MetaLDA: inference on new documents");
        CommandOption.process (MetaLDAInfer.class, args);





        if(testDocInputFile.value != null)
        {

            InstanceList testDocs = InstanceList.load(new File(testDocInputFile.value));

            MatFileReader matReader = null;
            try {
                matReader = new MatFileReader(saveFolderName.value + "/train_stats.mat");
            } catch (IOException e) {
                e.printStackTrace();
                System.err.println("Reading training info error!");
                System.exit(-1);
            }

            Alphabet trainAlphabet = null;

            Alphabet trainTargetAlphabet = null;

            try {
                trainAlphabet = AlphabetFactory.loadFromFile(new File(saveFolderName.value + "/train_alphabet.txt"));
                trainTargetAlphabet = AlphabetFactory.loadFromFile(new File(saveFolderName.value + "/train_target_alphabet.txt"));
            } catch (IOException e) {
                e.printStackTrace();
                System.err.println("Reading training alphabets error!");
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






            MetaLDAInfer inferencer = null;


            if (sampleAlphaMethod == 1)
            {
                inferencer = new MetaLDAInfer(trainAlphabet, trainTargetAlphabet, lambda, phi);
            }
            else
            {
                inferencer = new MetaLDAInfer(trainAlphabet, trainTargetAlphabet, alpha[0], phi);
            }

            inferencer.setRandomSeed(randomSeedOption.value);

            File testSaveFile = null;

            testSaveFile = new File(saveFolderName.value + "/test_stats.mat");


            MLDouble testThetaMat = null;

            double perplexity = 0.0;


            if (isComputePerplexity.value == true)
            {
                testThetaMat = new MLDouble("test_theta", inferencer.getInferredDistributions(testDocs, numInferenceIterationsOption.value, 10, 10, true));
                perplexity = inferencer.computePerplexityForEverySecondWord(testDocs, testThetaMat.getArray(), phi);
            }
            else
            {
                testThetaMat = new MLDouble("test_theta", inferencer.getInferredDistributions(testDocs, numInferenceIterationsOption.value, 10, 10, false));
            }


            ArrayList saveList = new ArrayList();

            saveList.add(testThetaMat);


            if (sampleAlphaMethod == 1) {
                MLDouble testAlphaMat = new MLDouble("test_alpha", inferencer.alpha);
                saveList.add(testAlphaMat);
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
//            testVocOut.println(testDocs.getAlphabet().toString());
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
//            testLabelOut.println(testDocs.getTargetAlphabet().toString());
//
//            testLabelOut.close();
        }








    }



}