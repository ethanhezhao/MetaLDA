

package topicmodels;

import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.*;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.Randoms;
import com.carrotsearch.hppc.ObjectIntHashMap;

import java.io.*;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/**
 * Simple parallel threaded implementation of LDA,
 *  following Newman, Asuncion, Smyth and Welling, Distributed Algorithms for Topic Models
 *  JMLR (2009), with SparseLDA sampling scheme and data structure from
 *  Yao, Mimno and McCallum, Efficient Methods for Topic Model Inference on Streaming Document Collections, KDD (2009).
 * 
 * @author David Mimno, Andrew McCallum
 */

public abstract class ParallelTopicModelHyper implements Serializable {

	public static final int UNASSIGNED_TOPIC = -1;

	public static Logger logger = MalletLogger.getLogger(ParallelTopicModelHyper.class.getName());
	
	public ArrayList<TopicAssignment> data;  // the training instances and their topic assignments
	public Alphabet alphabet; // the alphabet for the input data
	public LabelAlphabet topicAlphabet;  // the alphabet for the topics
	
	public int numTopics; // Number of topics to be fit

	// These values are used to encode type/topic counts as
	//  count/topic pairs in a single int.
	public int topicMask;
	public int topicBits;

	public int numTypes;
	public int totalTokens;

	
	public double[][] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	public double[] alphaSum;
	public double commonAlphaSum;

	public double commonBeta;
	public double[][] beta;   // Prior on per-topic multinomial distribution over words
	public double[] betaSum;

	public static final double DEFAULT_BETA = 0.01;
	
	public int[][] typeTopicCounts; // indexed by <feature index, topic index>
	public int[] tokensPerTopic; // indexed by <topic index>


	public int numIterations = 1000;
	public int burninPeriod = 200; 
	public int saveSampleInterval = 10; 
	public int optimizeAlphaInterval = 50;
	public int optimizeBetaInterval = 10;

	public int showTopicsInterval = 50;
	public int wordsPerTopic = 7;

	
	public int randomSeed = -1;
	public NumberFormat formatter;
	public boolean printLogLikelihood = false;


	/*if true, beta is a full matrix, then "document only" bucket can not be cached. So no sparse LDA framework*/
	protected boolean isFullBeta = true;

	// The number of times each type appears in the corpus
	int[] typeTotals;
	// The max over typeTotals, used for beta optimization
	int maxTypeCount; 
	
	int numThreads = 1;
	
	public ParallelTopicModelHyper (int numberOfTopics) {
		this (numberOfTopics, numberOfTopics, DEFAULT_BETA);
	}
	
	public ParallelTopicModelHyper (int numberOfTopics, double alphaSum, double beta) {
		this (newLabelAlphabet (numberOfTopics), alphaSum, beta);
	}
	
	private static LabelAlphabet newLabelAlphabet (int numTopics) {
		LabelAlphabet ret = new LabelAlphabet();
		for (int i = 0; i < numTopics; i++)
			ret.lookupIndex("topic"+i);
		return ret;
	}

	public ParallelTopicModelHyper (LabelAlphabet topicAlphabet, double alphaSum, double beta) {
		
		this.data = new ArrayList<TopicAssignment>();
		this.topicAlphabet = topicAlphabet;
		this.commonAlphaSum = alphaSum;

		this.commonBeta = beta;

		setNumTopics(topicAlphabet.size());

		formatter = NumberFormat.getInstance();
		formatter.setMaximumFractionDigits(5);

		logger.info("Mallet LDA: " + numTopics + " topics, " + topicBits + " topic bits, " +
					Integer.toBinaryString(topicMask) + " topic mask");
	}
	
	public Alphabet getAlphabet() { return alphabet; }
	public LabelAlphabet getTopicAlphabet() { return topicAlphabet; }
	public int getNumTopics() { return numTopics; }
	
	/** Set or reset the number of topics. This method will not change any token-topic assignments,
		so it should only be used before initializing or restoring a previously saved state. */
	public void setNumTopics(int numTopics) {
		this.numTopics = numTopics;

		if (Integer.bitCount(numTopics) == 1) {
			// exact power of 2
			topicMask = numTopics - 1;
			topicBits = Integer.bitCount(topicMask);
		}
		else {
			// otherwise add an extra bit
			topicMask = Integer.highestOneBit(numTopics) * 2 - 1;
			topicBits = Integer.bitCount(topicMask);
		}
		

		
		tokensPerTopic = new int[numTopics];
	}

	public ArrayList<TopicAssignment> getData() { return data; }
	
	public int[][] getTypeTopicCounts() { return typeTopicCounts; }
	public int[] getTokensPerTopic() { return tokensPerTopic; }

	public void setNumIterations (int numIterations) {
		this.numIterations = numIterations;
	}

	public void setBurninPeriod (int burninPeriod) {
		this.burninPeriod = burninPeriod;
	}

	public void setTopicDisplay(int interval, int n) {
		this.showTopicsInterval = interval;
		this.wordsPerTopic = n;
	}

	public void setRandomSeed(int seed) {
		randomSeed = seed;
	}

	/** Interval for optimizing Dirichlet hyperparameters */
	public void setOptimizeInterval(int alphaInterval, int betaInterval) {

		this.optimizeAlphaInterval = alphaInterval;

		this.optimizeBetaInterval = betaInterval;
//
//		// Make sure we always have at least one sample
//		//  before optimizing hyperparameters
//		if (saveSampleInterval > optimizeInterval) {
//			saveSampleInterval = optimizeInterval;
//		}
	}


	public void setNumThreads(int threads) {
		this.numThreads = threads;
	}


	public void addInstances (InstanceList training) {

		alphabet = training.getDataAlphabet();
		numTypes = alphabet.size();

		Randoms random = null;
		if (randomSeed == -1) {
			random = new Randoms();
		}
		else {
			random = new Randoms(randomSeed);
		}

		for (Instance instance : training) {
			FeatureSequence tokens = (FeatureSequence) instance.getData();
			LabelSequence topicSequence =
				new LabelSequence(topicAlphabet, new int[ tokens.size() ]);

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < topics.length; position++) {

				int topic = random.nextInt(numTopics);
				topics[position] = topic;

			}

			TopicAssignment t = new TopicAssignment(instance, topicSequence);
			data.add(t);
		}

		this.alphaSum = new double[data.size()];

		Arrays.fill(this.alphaSum, this.commonAlphaSum);

		this.alpha = new double[this.data.size()][];
		for(int doc = 0; doc < this.data.size(); doc++)
		{
			alpha[doc] = new double[numTopics];
			Arrays.fill(alpha[doc], alphaSum[doc] / numTopics);

		}


		this.betaSum = new double[this.numTopics];

		this.beta = new double[numTopics][];
		for(int topic = 0; topic < this.numTopics; topic++)
		{
			beta[topic] = new double[this.numTypes];
			Arrays.fill(beta[topic], this.commonBeta);

			this.betaSum[topic] = this.commonBeta * numTypes;

		}

		buildInitialTypeTopicCounts();

		logger.info("num docs: " + data.size());

		logger.info("num types: " + numTypes);


	}


	public void buildInitialTypeTopicCounts () {
		
		typeTopicCounts = new int[numTypes][];
		tokensPerTopic = new int[numTopics];

		// Get the total number of occurrences of each word type
		//int[] typeTotals = new int[numTrainTypes];
		typeTotals = new int[numTypes];
		
		// Create the type-topic counts data structure
		for (TopicAssignment document : data) {

			FeatureSequence tokens = (FeatureSequence) document.instance.getData();
			for (int position = 0; position < tokens.getLength(); position++) {
				int type = tokens.getIndexAtPosition(position);
				typeTotals[ type ]++;
			}
		}

		maxTypeCount = 0;

		// Allocate enough space so that we never have to worry about
		//  overflows: either the number of topics or the number of times
		//  the type occurs.
		for (int type = 0; type < numTypes; type++) {
			if (typeTotals[type] > maxTypeCount) { maxTypeCount = typeTotals[type]; }
			typeTopicCounts[type] = new int[ Math.min(numTopics, typeTotals[type]) ];
		}
		
		for (TopicAssignment document : data) {

			FeatureSequence tokens = (FeatureSequence) document.instance.getData();
			FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;

			int[] topics = topicSequence.getFeatures();
			for (int position = 0; position < tokens.size(); position++) {

				int topic = topics[position];
				
				if (topic == UNASSIGNED_TOPIC) { continue; }

				tokensPerTopic[topic]++;
				
				// The format for these arrays is 
				//  the topic in the rightmost bits
				//  the count in the remaining (left) bits.
				// Since the count is in the high bits, sorting (desc)
				//  by the numeric value of the int guarantees that
				//  higher counts will be before the lower counts.
				
				int type = tokens.getIndexAtPosition(position);
				int[] currentTypeTopicCounts = typeTopicCounts[ type ];
		
				// Start by assuming that the array is either empty
				//  or is in sorted (descending) order.
				
				// Here we are only adding counts, so if we find 
				//  an existing location with the topic, we only need
				//  to ensure that it is not larger than its left neighbor.
				
				int index = 0;
				int currentTopic = currentTypeTopicCounts[index] & topicMask;
				int currentValue;
				
				while (currentTypeTopicCounts[index] > 0 && currentTopic != topic) {
					index++;
					if (index == currentTypeTopicCounts.length) {
						logger.info("overflow on type " + type);
					}
					currentTopic = currentTypeTopicCounts[index] & topicMask;
				}
				currentValue = currentTypeTopicCounts[index] >> topicBits;
				
				if (currentValue == 0) {
					// new value is 1, so we don't have to worry about sorting
					//  (except by topic suffix, which doesn't matter)
					
					currentTypeTopicCounts[index] =
						(1 << topicBits) + topic;
				}
				else {
					currentTypeTopicCounts[index] =
						((currentValue + 1) << topicBits) + topic;
					
					// Now ensure that the array is still sorted by 
					//  bubbling this value up.
					while (index > 0 &&
						   currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
						int temp = currentTypeTopicCounts[index];
						currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
						currentTypeTopicCounts[index - 1] = temp;
						
						index--;
					}
				}
			}
		}
	}
	

	public void sumTypeTopicCounts (WorkerRunnableHyper[] runnables) {

		// Clear the topic totals
		Arrays.fill(tokensPerTopic, 0);
		
		// Clear the type/topic counts, only 
		//  looking at the entries before the first 0 entry.

		for (int type = 0; type < numTypes; type++) {
			
			int[] targetCounts = typeTopicCounts[type];
			
			int position = 0;
			while (position < targetCounts.length && 
				   targetCounts[position] > 0) {
				targetCounts[position] = 0;
				position++;
			}

		}

		for (int thread = 0; thread < numThreads; thread++) {

			// Handle the total-tokens-per-topic array

			int[] sourceTotals = runnables[thread].getTokensPerTopic();
			for (int topic = 0; topic < numTopics; topic++) {
				tokensPerTopic[topic] += sourceTotals[topic];
			}
			
			// Now handle the individual type topic counts
			
			int[][] sourceTypeTopicCounts = 
				runnables[thread].getTypeTopicCounts();
			
			for (int type = 0; type < numTypes; type++) {

				// Here the source is the individual thread counts,
				//  and the target is the global counts.

				int[] sourceCounts = sourceTypeTopicCounts[type];
				int[] targetCounts = typeTopicCounts[type];

				int sourceIndex = 0;
				while (sourceIndex < sourceCounts.length &&
					   sourceCounts[sourceIndex] > 0) {
					
					int topic = sourceCounts[sourceIndex] & topicMask;
					int count = sourceCounts[sourceIndex] >> topicBits;

					int targetIndex = 0;
					int currentTopic = targetCounts[targetIndex] & topicMask;
					int currentCount;
					
					while (targetCounts[targetIndex] > 0 && currentTopic != topic) {
						targetIndex++;
						if (targetIndex == targetCounts.length) {
							logger.info("overflow in merging on type " + type);
						}
						currentTopic = targetCounts[targetIndex] & topicMask;
					}
					currentCount = targetCounts[targetIndex] >> topicBits;
					
					targetCounts[targetIndex] =
						((currentCount + count) << topicBits) + topic;
					
					// Now ensure that the array is still sorted by 
					//  bubbling this value up.
					while (targetIndex > 0 &&
						   targetCounts[targetIndex] > targetCounts[targetIndex - 1]) {
						int temp = targetCounts[targetIndex];
						targetCounts[targetIndex] = targetCounts[targetIndex - 1];
						targetCounts[targetIndex - 1] = temp;
						
						targetIndex--;
					}
					
					sourceIndex++;
				}
				
			}
		}

		/* // Debuggging code to ensure counts are being 
		   // reconstructed correctly.

		for (int type = 0; type < numTrainTypes; type++) {
			
			int[] targetCounts = typeLabelTopicCounts[type];
			
			int index = 0;
			int count = 0;
			while (index < targetCounts.length &&
				   targetCounts[index] > 0) {
				count += targetCounts[index] >> topicBits;
				index++;
			}
			
			if (count != typeTotals[type]) {
				System.err.println("Expected " + typeTotals[type] + ", found " + count);
			}
			
		}
		*/
	}
	

	
	public abstract void optimizeAlpha(WorkerRunnableHyper[] runnables);


	public abstract void optimizeBeta(WorkerRunnableHyper[] runnables);

	public void estimate () throws IOException {

		long startTime = System.currentTimeMillis();

		WorkerRunnableHyper[] runnables = new WorkerRunnableHyper[numThreads];

		int docsPerThread = data.size() / numThreads;
		int offset = 0;

		if (numThreads > 1) {
		
			for (int thread = 0; thread < numThreads; thread++) {
				int[] runnableTotals = new int[numTopics];
				System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);
				
				int[][] runnableCounts = new int[numTypes][];
				for (int type = 0; type < numTypes; type++) {
					int[] counts = new int[typeTopicCounts[type].length];
					System.arraycopy(typeTopicCounts[type], 0, counts, 0, counts.length);
					runnableCounts[type] = counts;
				}
				
				// some docs may be missing at the end due to integer division
				if (thread == numThreads - 1) {
					docsPerThread = data.size() - offset;
				}
				
				Randoms random = null;
				if (randomSeed == -1) {
					random = new Randoms();
				}
				else {
					random = new Randoms(randomSeed);
				}

				runnables[thread] = new WorkerRunnableHyper(numTopics,
													   alpha, alphaSum, beta, betaSum,
													   random, data,
													   runnableCounts, runnableTotals,
													   offset, docsPerThread, isFullBeta);

				offset += docsPerThread;
			
			}
		}
		else {
			
			// If there is only one thread, copy the typeLabelTopicCounts
			//  arrays directly, rather than allocating new memory.

			Randoms random = null;
			if (randomSeed == -1) {
				random = new Randoms();
			}
			else {
				random = new Randoms(randomSeed);
			}

			runnables[0] = new WorkerRunnableHyper(numTopics,
											  alpha, alphaSum, beta, betaSum,
											  random, data,
											  typeTopicCounts, tokensPerTopic,
											  offset, docsPerThread, isFullBeta);

			// If there is only one thread, we 
			//  can avoid communications overhead.
			// This switch informs the thread not to 
			//  gather statistics for its portion of the data.
			runnables[0].makeOnlyThread();
		}

		ExecutorService executor = Executors.newFixedThreadPool(numThreads);

		long lastTime = 0;
	
		for (int iteration = 1; iteration <= numIterations; iteration++) {

			long iterationStart = System.currentTimeMillis();

			if (showTopicsInterval != 0 && iteration != 0 && iteration % showTopicsInterval == 0) {
				logger.info("\n" + displayTopWords (wordsPerTopic, false));
			}


			if (numThreads > 1) {
			
				// Submit runnables to thread pool
				
				for (int thread = 0; thread < numThreads; thread++) {
					
					logger.fine("submitting thread " + thread);
					executor.submit(runnables[thread]);
					//runnables[thread].run();
				}
				
				// I'm getting some problems that look like 
				//  a thread hasn't started yet when it is first
				//  polled, so it appears to be finished. 
				// This only occurs in very short corpora.
				try {
					Thread.sleep(20);
				} catch (InterruptedException e) {
					
				}
				
				boolean finished = false;
				while (! finished) {
					
					try {
						Thread.sleep(10);
					} catch (InterruptedException e) {
						
					}
					
					finished = true;
					
					// Are all the threads done?
					for (int thread = 0; thread < numThreads; thread++) {
						//logger.info("thread " + thread + " done? " + runnables[thread].isFinished);
						finished = finished && runnables[thread].isFinished;
					}
					
				}
				
				//System.out.print("[" + (System.currentTimeMillis() - iterationStart) + "] ");
				
				sumTypeTopicCounts(runnables);
				
				//System.out.print("[" + (System.currentTimeMillis() - iterationStart) + "] ");
				
				for (int thread = 0; thread < numThreads; thread++) {
					int[] runnableTotals = runnables[thread].getTokensPerTopic();
					System.arraycopy(tokensPerTopic, 0, runnableTotals, 0, numTopics);
					
					int[][] runnableCounts = runnables[thread].getTypeTopicCounts();
					for (int type = 0; type < numTypes; type++) {
						int[] targetCounts = runnableCounts[type];
						int[] sourceCounts = typeTopicCounts[type];
						
						int index = 0;
						while (index < sourceCounts.length) {
							
							if (sourceCounts[index] != 0) {
								targetCounts[index] = sourceCounts[index];
							}
							else if (targetCounts[index] != 0) {
								targetCounts[index] = 0;
							}
							else {
								break;
							}
							
							index++;
						}
						//System.arraycopy(typeLabelTopicCounts[type], 0, counts, 0, counts.length);
					}
				}
			}
			else {
				runnables[0].run();
			}

			long elapsedMillis = System.currentTimeMillis() - iterationStart;
			if (elapsedMillis < 1000) {
				logger.fine(elapsedMillis + "ms ");
			}
			else {
				logger.fine((elapsedMillis/1000) + "s ");
			}   

			if (iteration > burninPeriod && optimizeAlphaInterval != 0 &&
				iteration % optimizeAlphaInterval == 0) {

				optimizeAlpha(runnables);

//				logger.info("interation: " + iteration);
				
				logger.fine("[O " + (System.currentTimeMillis() - iterationStart) + "] ");
			}


			if (iteration > burninPeriod && optimizeBetaInterval != 0 &&
					iteration % optimizeBetaInterval == 0) {

				optimizeBeta(runnables);

//				logger.info("interation: " + iteration);

				logger.fine("[O " + (System.currentTimeMillis() - iterationStart) + "] ");
			}
			
			if (iteration % 10 == 0) {
				if (printLogLikelihood) {
					logger.fine ("<" + iteration + "> LL/token: " + formatter.format(perplexity()));

                }
			}

			long nowTime = System.currentTimeMillis();
			double during = (double)(nowTime - lastTime)/1000;

			if (iteration % 10 == 0)
				logger.info ("<" + iteration + "> in " + formatter.format(during) + " seconds");
			lastTime = nowTime;


		}

		executor.shutdownNow();
	
		long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
		long minutes = seconds / 60;	seconds %= 60;
		long hours = minutes / 60;	minutes %= 60;
		long days = hours / 24;	hours %= 24;

		StringBuilder timeReport = new StringBuilder();
		timeReport.append("\nTotal time: ");
		if (days != 0) { timeReport.append(days); timeReport.append(" days "); }
		if (hours != 0) { timeReport.append(hours); timeReport.append(" hours "); }
		if (minutes != 0) { timeReport.append(minutes); timeReport.append(" minutes "); }
		timeReport.append(seconds); timeReport.append(" seconds");
		
		logger.info(timeReport.toString());
	}


	/**
	 *  Return an array of sorted sets (one set per topic). Each set 
	 *   contains IDSorter objects with integer keys into the alphabet.
	 *   To get direct access to the Strings, use getTopWords().
	 */
	public ArrayList<TreeSet<IDSorter>> getSortedWords () {

		ArrayList<TreeSet<IDSorter>> topicSortedWords = new ArrayList<TreeSet<IDSorter>>(numTopics);

		// Initialize the tree sets
		for (int topic = 0; topic < numTopics; topic++) {
			topicSortedWords.add(new TreeSet<IDSorter>());
		}

		// Collect counts

		int[] topicIndicator = new int[numTopics];

		for (int type = 0; type < numTypes; type++) {

			Arrays.fill(topicIndicator, 0);

			int[] topicCounts = typeTopicCounts[type];

			int index = 0;
			while (index < topicCounts.length &&
				   topicCounts[index] > 0) {

				int topic = topicCounts[index] & topicMask;
				int count = topicCounts[index] >> topicBits;

				topicSortedWords.get(topic).add(new IDSorter(type, count + beta[topic][type]));

				topicIndicator[topic] = 1;

				index++;
			}
			for(int topic = 0; topic  < numTopics; topic ++)
			{
				if (topicIndicator[topic] == 0)
				{
					topicSortedWords.get(topic).add(new IDSorter(type, beta[topic][type]));
				}
			}
		}

		return topicSortedWords;
	}

	/** Return an array (one element for each topic) of arrays of words, which
	 *  are the most probable words for that topic in descending order. These
	 *  are returned as Objects, but will probably be Strings.
	 *
	 *  @param numWords The maximum length of each topic's array of words (may be less).
	 */
	
	public Object[][] getTopWords(int numWords) {

		ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();
		Object[][] result = new Object[ numTopics ][];

		for (int topic = 0; topic < numTopics; topic++) {
			
			TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
			
			// How many words should we report? Some topics may have fewer than
			//  the default number of words with non-zero weight.
			int limit = numWords;
			if (sortedWords.size() < numWords) { limit = sortedWords.size(); }

			result[topic] = new Object[limit];

			Iterator<IDSorter> iterator = sortedWords.iterator();
			for (int i=0; i < limit; i++) {
				IDSorter info = iterator.next();
				result[topic][i] = alphabet.lookupObject(info.getID());
			}
		}

		return result;
	}

	public void printTopWords (File file, int numWords, boolean useNewLines) throws IOException {
		PrintStream out = new PrintStream (file);
		printTopWords(out, numWords, useNewLines);
		out.close();
	}
	
	public void printTopWords (PrintStream out, int numWords, boolean usingNewLines) {
		out.print(displayTopWords(numWords, usingNewLines));
	}



	public String displayTopWords (int numWords, boolean usingNewLines) {

		StringBuilder out = new StringBuilder();

		ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();

		// Print results for each topic
		for (int topic = 0; topic < numTopics; topic++) {
			TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
			int word = 0;
			Iterator<IDSorter> iterator = sortedWords.iterator();

			if (usingNewLines) {
				while (iterator.hasNext() && word < numWords) {
					IDSorter info = iterator.next();
					out.append(alphabet.lookupObject(info.getID()) + "\t" + formatter.format(info.getWeight()) + "\n");
					word++;
				}
			}
			else {

				while (iterator.hasNext() && word < numWords) {
					IDSorter info = iterator.next();
					out.append(alphabet.lookupObject(info.getID()) + " ");
					word++;
				}
				out.append ("\n");
			}
		}

		return out.toString();
	}
	
	public void topicXMLReport (PrintWriter out, int numWords) {
		ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();
		
		out.println("<?xml version='1.0' ?>");
		out.println("<topicModel>");
		for (int topic = 0; topic < numTopics; topic++) {
			out.println("  <topic id='" + topic + "' alpha='" + alpha[topic] +
						"' totalTokens='" + tokensPerTopic[topic] + "'>");
			int rank = 1;
			Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();
			while (iterator.hasNext() && rank <= numWords) {
				IDSorter info = iterator.next();
				out.println("	<word rank='" + rank + "' count='" + info.getWeight() + "'>" +
						  alphabet.lookupObject(info.getID()) + 
						  "</word>");
				rank++;
			}
			out.println("  </topic>");
		}
		out.println("</topicModel>");
	}

	public void topicPhraseXMLReport(PrintWriter out, int numWords) {
		int numTopics = this.getNumTopics();
		ObjectIntHashMap<String>[] phrases = new ObjectIntHashMap[numTopics];
		Alphabet alphabet = this.getAlphabet();
		
		// Get counts of phrases
		for (int ti = 0; ti < numTopics; ti++)
			phrases[ti] = new ObjectIntHashMap<String>();
		for (int di = 0; di < this.getData().size(); di++) {
			TopicAssignment t = this.getData().get(di);
			Instance instance = t.instance;
			FeatureSequence fvs = (FeatureSequence) instance.getData();
			boolean withBigrams = false;
			if (fvs instanceof FeatureSequenceWithBigrams) withBigrams = true;
			int prevtopic = -1;
			int prevfeature = -1;
			int topic = -1;
			StringBuffer sb = null;
			int feature = -1;
			int doclen = fvs.size();
			for (int pi = 0; pi < doclen; pi++) {
				feature = fvs.getIndexAtPosition(pi);
				topic = this.getData().get(di).topicSequence.getIndexAtPosition(pi);
				if (topic == prevtopic && (!withBigrams || ((FeatureSequenceWithBigrams)fvs).getBiIndexAtPosition(pi) != -1)) {
					if (sb == null)
						sb = new StringBuffer (alphabet.lookupObject(prevfeature).toString() + " " + alphabet.lookupObject(feature));
					else {
						sb.append (" ");
						sb.append (alphabet.lookupObject(feature));
					}
				} else if (sb != null) {
					String sbs = sb.toString();
					//logger.info ("phrase:"+sbs);
					if (phrases[prevtopic].get(sbs) == 0)
						phrases[prevtopic].put(sbs,0);
					phrases[prevtopic].addTo(sbs, 1);
					prevtopic = prevfeature = -1;
					sb = null;
				} else {
					prevtopic = topic;
					prevfeature = feature;
				}
			}
		}
		// phrases[] now filled with counts
		
		// Now start printing the XML
		out.println("<?xml version='1.0' ?>");
		out.println("<topics>");

		ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();
		double[] probs = new double[alphabet.size()];
		for (int ti = 0; ti < numTopics; ti++) {
			out.print("  <topic id=\"" + ti + "\" alpha=\"" + alpha[ti] +
					"\" totalTokens=\"" + tokensPerTopic[ti] + "\" ");

			// For gathering <term> and <phrase> output temporarily 
			// so that we can get topic-title information before printing it to "out".
			ByteArrayOutputStream bout = new ByteArrayOutputStream();
			PrintStream pout = new PrintStream (bout);
			// For holding candidate topic titles
			AugmentableFeatureVector titles = new AugmentableFeatureVector (new Alphabet());

			// Print words
			int word = 0;
			Iterator<IDSorter> iterator = topicSortedWords.get(ti).iterator();
			while (iterator.hasNext() && word < numWords) {
				IDSorter info = iterator.next();
				pout.println("	<word weight=\""+(info.getWeight()/tokensPerTopic[ti])+"\" count=\""+Math.round(info.getWeight())+"\">"
							+ alphabet.lookupObject(info.getID()) +
						  "</word>");
				word++;
				if (word < 20) // consider top 20 individual words as candidate titles
					titles.add(alphabet.lookupObject(info.getID()), info.getWeight());
			}

			/*
			for (int type = 0; type < alphabet.size(); type++)
				probs[type] = this.getCountFeatureTopic(type, ti) / (double)this.getCountTokensPerTopic(ti);
			RankedFeatureVector rfv = new RankedFeatureVector (alphabet, probs);
			for (int ri = 0; ri < numWords; ri++) {
				int fi = rfv.getIndexAtRank(ri);
				pout.println ("	  <term weight=\""+probs[fi]+"\" count=\""+this.getCountFeatureTopic(fi,ti)+"\">"+alphabet.lookupObject(fi)+	"</term>");
				if (ri < 20) // consider top 20 individual words as candidate titles
					titles.add(alphabet.lookupObject(fi), this.getCountFeatureTopic(fi,ti));
			}
			*/

			// Print phrases
			Object[] keys = phrases[ti].keys().toArray();
			int[] values = phrases[ti].values().toArray();
			double counts[] = new double[keys.length];
			for (int i = 0; i < counts.length; i++)	counts[i] = values[i];
			double countssum = MatrixOps.sum (counts);	
			Alphabet alph = new Alphabet(keys);
			RankedFeatureVector rfv = new RankedFeatureVector (alph, counts);
			int max = rfv.numLocations() < numWords ? rfv.numLocations() : numWords;
			for (int ri = 0; ri < max; ri++) {
				int fi = rfv.getIndexAtRank(ri);
				pout.println ("	<phrase weight=\""+counts[fi]/countssum+"\" count=\""+values[fi]+"\">"+alph.lookupObject(fi)+	"</phrase>");
				// Any phrase count less than 20 is simply unreliable
				if (ri < 20 && values[fi] > 20) 
					titles.add(alph.lookupObject(fi), 100*values[fi]); // prefer phrases with a factor of 100 
			}
			
			// Select candidate titles
			StringBuffer titlesStringBuffer = new StringBuffer();
			rfv = new RankedFeatureVector (titles.getAlphabet(), titles);
			int numTitles = 10; 
			for (int ri = 0; ri < numTitles && ri < rfv.numLocations(); ri++) {
				// Don't add redundant titles
				if (titlesStringBuffer.indexOf(rfv.getObjectAtRank(ri).toString()) == -1) {
					titlesStringBuffer.append (rfv.getObjectAtRank(ri));
					if (ri < numTitles-1)
						titlesStringBuffer.append (", ");
				} else
					numTitles++;
			}
			out.println("titles=\"" + titlesStringBuffer.toString() + "\">");
			out.print(bout.toString());
			out.println("  </topic>");
		}
		out.println("</topics>");
	}

	

	/**
	 *  Write the internal representation of type-topic counts  
	 *   (count/topic pairs in descending order by count) to a file.
	 */
	public void printTypeTopicCounts(PrintStream out) throws IOException {
//		PrintWriter out = new PrintWriter (new FileWriter (file) );

		for (int type = 0; type < numTypes; type++) {

			StringBuilder buffer = new StringBuilder();

			buffer.append(type + " " + alphabet.lookupObject(type));

			int[] topicCounts = typeTopicCounts[type];

			int index = 0;
			while (index < topicCounts.length &&
				   topicCounts[index] > 0) {

				int topic = topicCounts[index] & topicMask;
				int count = topicCounts[index] >> topicBits;
				
				buffer.append(" " + topic + ":" + count);

				index++;
			}

			out.println(buffer);
		}

		out.close();
	}

//	public void printTopicWordWeights(File file) throws IOException {
//		PrintWriter out = new PrintWriter (new FileWriter (file) );
//		printTopicWordWeights(out);
//		out.close();
//	}

	/**
	 * Print an unnormalized weight for every word in every topic.
	 *  Most of these will be equal to the smoothing parameter beta.
	 */
	public void printTopicWordWeights(PrintStream out) throws IOException {
		// Probably not the most efficient way to do this...

		for (int topic = 0; topic < numTopics; topic++) {
			for (int type = 0; type < numTypes; type++) {

				int[] topicCounts = typeTopicCounts[type];
				
				double weight = beta[topic][type];

				int index = 0;
				while (index < topicCounts.length &&
					   topicCounts[index] > 0) {

					int currentTopic = topicCounts[index] & topicMask;
					
					
					if (currentTopic == topic) {
						weight += topicCounts[index] >> topicBits;
						break;
					}

					index++;
				}

				out.println(topic + "\t" + alphabet.lookupObject(type) + "\t" + weight);

			}
		}
	}
	
	/** Get the smoothed distribution over topics for a training instance. 
	 */
	public double[] getTopicProbabilities(int instanceID) {
		LabelSequence topics = data.get(instanceID).topicSequence;
		return getTopicProbabilities(instanceID, topics);
	}

	/** Get the smoothed distribution over topics for a topic sequence, 
	 * which may be from the training set or from a new instance with topics
	 * assigned by an inferencer.
	 */
	public double[] getTopicProbabilities(int doc, LabelSequence topics) {
		double[] topicDistribution = new double[numTopics];

		// Loop over the tokens in the document, counting the current topic
		//  assignments.
		for (int position = 0; position < topics.getLength(); position++) {
			topicDistribution[ topics.getIndexAtPosition(position) ]++;
		}

		// Add the smoothing parameters and normalize
		double sum = 0.0;
		for (int topic = 0; topic < numTopics; topic++) {
			topicDistribution[topic] += alpha[doc][topic];
			sum += topicDistribution[topic];
		}

		// And normalize
		for (int topic = 0; topic < numTopics; topic++) {
			topicDistribution[topic] /= sum;
		}

		return topicDistribution;
	}

//	public void printDocumentTopics (File file) throws IOException {
//		PrintWriter out = new PrintWriter (new FileWriter (file) );
//		printDocumentTopics (out);
//		out.close();
//	}

	public void printDenseDocumentTopics(PrintStream out) {
		int docLen;
		int[] topicCounts = new int[numTopics];
		for (int doc = 0; doc < data.size(); doc++) {
			LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
			int[] currentDocTopics = topicSequence.getFeatures();

			StringBuilder builder = new StringBuilder();

			builder.append(doc);
			builder.append("\t");

			if (data.get(doc).instance.getName() != null) {
				builder.append(data.get(doc).instance.getName()); 
			}
			else {
				builder.append("no-name");
			}

			docLen = currentDocTopics.length;

			// Count up the tokens
			for (int token=0; token < docLen; token++) {
				topicCounts[ currentDocTopics[token] ]++;
			}

			// And normalize
			for (int topic = 0; topic < numTopics; topic++) {
				builder.append("\t" + ((alpha[doc][topic] + topicCounts[topic]) / (docLen + alphaSum[doc]) ));
			}
			out.println(builder);

			Arrays.fill(topicCounts, 0);
		}		
	}

//	public void printDocumentTopics (PrintWriter out) {
//		printDocumentTopics (out, 0.0, -1);
//	}

	/**
	 *  @param out		  A print writer
	 *  @param threshold   Only print topics with proportion greater than this number
	 *  @param max		 Print no more than this many topics
	 */
	public void printDocumentTopics (PrintStream out, double threshold, int max)	{
		out.print ("#doc name topic proportion ...\n");
		int docLen;
		int[] topicCounts = new int[ numTopics ];

		IDSorter[] sortedTopics = new IDSorter[ numTopics ];
		for (int topic = 0; topic < numTopics; topic++) {
			// Initialize the sorters with dummy values
			sortedTopics[topic] = new IDSorter(topic, topic);
		}

		if (max < 0 || max > numTopics) {
			max = numTopics;
		}

		for (int doc = 0; doc < data.size(); doc++) {
			LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
			int[] currentDocTopics = topicSequence.getFeatures();

			StringBuilder builder = new StringBuilder();

			builder.append(doc);
			builder.append("\t");

			if (data.get(doc).instance.getName() != null) {
				builder.append(data.get(doc).instance.getName()); 
			}
			else {
				builder.append("no-name");
			}

			builder.append("\t");
			docLen = currentDocTopics.length;

			// Count up the tokens
			for (int token=0; token < docLen; token++) {
				topicCounts[ currentDocTopics[token] ]++;
			}

			// And normalize
			for (int topic = 0; topic < numTopics; topic++) {
				sortedTopics[topic].set(topic, (alpha[doc][topic] + topicCounts[topic]) / (docLen + alphaSum[doc]) );
			}
			
			Arrays.sort(sortedTopics);

			for (int i = 0; i < max; i++) {
				if (sortedTopics[i].getWeight() < threshold) { break; }
				
				builder.append(sortedTopics[i].getID() + "\t" + 
							   sortedTopics[i].getWeight() + "\t");
			}
			out.println(builder);

			Arrays.fill(topicCounts, 0);
		}
		
	}
	
	public double[][] getSubCorpusTopicWords(boolean[] documentMask, boolean normalized, boolean smoothed) {		
		double[][] result = new double[numTopics][numTypes];
		int[] subCorpusTokensPerTopic = new int[numTopics];
		
		for (int doc = 0; doc < data.size(); doc++) {
			if (documentMask[doc]) {
				int[] words = ((FeatureSequence) data.get(doc).instance.getData()).getFeatures();
				int[] topics = data.get(doc).topicSequence.getFeatures();
				for (int position = 0; position < topics.length; position++) {
					result[ topics[position] ][ words[position] ]++;
					subCorpusTokensPerTopic[ topics[position] ]++;
				}
			}
		}

		if (smoothed) {
			for (int topic = 0; topic < numTopics; topic++) {
				for (int type = 0; type < numTypes; type++) {
					result[topic][type] += beta[topic][type];
				}
			}
		}

		if (normalized) {
			double[] topicNormalizers = new double[numTopics];
			if (smoothed) {
				for (int topic = 0; topic < numTopics; topic++) {
					topicNormalizers[topic] = 1.0 / (subCorpusTokensPerTopic[topic] + betaSum[topic]);
				}
			}
			else {
				for (int topic = 0; topic < numTopics; topic++) {
					topicNormalizers[topic] = 1.0 / subCorpusTokensPerTopic[topic];
				}
			}

			for (int topic = 0; topic < numTopics; topic++) {
				for (int type = 0; type < numTypes; type++) {
					result[topic][type] *= topicNormalizers[topic];
				}
			}
		}

		return result;
	}

	public double[][] getTopicWords(boolean normalized, boolean smoothed) {
		double[][] result = new double[numTopics][numTypes];

		for (int type = 0; type < numTypes; type++) {
			int[] topicCounts = typeTopicCounts[type];

			int index = 0;
			while (index < topicCounts.length &&
				   topicCounts[index] > 0) {

				int topic = topicCounts[index] & topicMask;
				int count = topicCounts[index] >> topicBits;

				result[topic][type] += count;

				index++;
			}
		}

		if (smoothed) {
			for (int topic = 0; topic < numTopics; topic++) {
				for (int type = 0; type < numTypes; type++) {
					result[topic][type] += beta[topic][type];
				}
			}
		}

		if (normalized) {
			double[] topicNormalizers = new double[numTopics];
			if (smoothed) {
				for (int topic = 0; topic < numTopics; topic++) {
					topicNormalizers[topic] = 1.0 / (tokensPerTopic[topic] + betaSum[topic]);
				}
			}
			else {
				for (int topic = 0; topic < numTopics; topic++) {
					topicNormalizers[topic] = 1.0 / tokensPerTopic[topic];
				}
			}

			for (int topic = 0; topic < numTopics; topic++) {
				for (int type = 0; type < numTypes; type++) {
					result[topic][type] *= topicNormalizers[topic];
				}
			}
		}

		return result;
	}

	public double[][] getDocumentTopics(boolean normalized, boolean smoothed) {
		double[][] result = new double[data.size()][numTopics];

		for (int doc = 0; doc < data.size(); doc++) {
			int[] topics = data.get(doc).topicSequence.getFeatures();
			for (int position = 0; position < topics.length; position++) {
				result[doc][ topics[position] ]++;
			}

			if (smoothed) {
				for (int topic = 0; topic < numTopics; topic++) {
					result[doc][topic] += alpha[doc][topic];
				}
			}

			if (normalized) {
				double sum = 0.0;
				for (int topic = 0; topic < numTopics; topic++) {
					sum += result[doc][topic];
				}
				double normalizer = 1.0 / sum;
				for (int topic = 0; topic < numTopics; topic++) {
					result[doc][topic] *= normalizer;
				}				
			}
		}

		return result;
	}
	
	public ArrayList<TreeSet<IDSorter>> getTopicDocuments(double smoothing) {
		ArrayList<TreeSet<IDSorter>> topicSortedDocuments = new ArrayList<TreeSet<IDSorter>>(numTopics);

		// Initialize the tree sets
		for (int topic = 0; topic < numTopics; topic++) {
			topicSortedDocuments.add(new TreeSet<IDSorter>());
		}

		int[] topicCounts = new int[numTopics];

		for (int doc = 0; doc < data.size(); doc++) {
			int[] topics = data.get(doc).topicSequence.getFeatures();
			for (int position = 0; position < topics.length; position++) {
				topicCounts[ topics[position] ]++;
			}

			for (int topic = 0; topic < numTopics; topic++) {
				topicSortedDocuments.get(topic).add(new IDSorter(doc, (topicCounts[topic] + smoothing) / (topics.length + numTopics * smoothing) ));
				topicCounts[topic] = 0;
			}
		}

		return topicSortedDocuments;
	}
	
	public void printTopicDocuments (PrintWriter out) {
		printTopicDocuments (out, 100);
	}

//	/**
//	 *  @param out		  A print writer
//	 *  @param count      Print this number of top documents
//	 */
	public void printTopicDocuments (PrintWriter out, int max)	{
		out.println("#topic doc name proportion ...");
		
		ArrayList<TreeSet<IDSorter>> topicSortedDocuments = getTopicDocuments(10.0);

		for (int topic = 0; topic < numTopics; topic++) {
			TreeSet<IDSorter> sortedDocuments = topicSortedDocuments.get(topic);
			
			int i = 0;
			for (IDSorter sorter: sortedDocuments) {
				if (i == max) { break; }
				
				int doc = sorter.getID();
				double proportion = sorter.getWeight();
				String name = data.get(doc).instance.getName().toString();
				if (name == null) {
					name = "no-name";
				}
				out.format("%d %d %s %f\n", topic, doc, name, proportion);
				
				i++;
			}
		}
	}



//	public void printState (File f) throws IOException {
//		PrintStream out =
//			new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(f))));
//		printState(out);
//		out.close();
//	}
//
//	public void printState (PrintStream out) {
//
//		out.println ("#doc source pos typeindex type topic");
//		out.print("#alpha : ");
//		for (int topic = 0; topic < numTopics; topic++) {
//			out.print(alpha[topic] + " ");
//		}
//		out.println();
//		out.println("#beta : " + beta);
//
//		for (int doc = 0; doc < data.size(); doc++) {
//			FeatureSequence tokenSequence =	(FeatureSequence) data.get(doc).instance.getData();
//			LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;
//
//			String source = "NA";
//			if (data.get(doc).instance.getSource() != null) {
//				source = data.get(doc).instance.getSource().toString();
//			}
//
//			Formatter output = new Formatter(new StringBuilder(), Locale.US);
//
//			for (int pi = 0; pi < topicSequence.getLength(); pi++) {
//				int type = tokenSequence.getIndexAtPosition(pi);
//				int topic = topicSequence.getIndexAtPosition(pi);
//
//				output.format("%d %s %d %d %s %d\n", doc, source, pi, type, alphabet.lookupObject(type), topic);
//
//				/*
//				out.print(doc); out.print(' ');
//				out.print(source); out.print(' ');
//				out.print(pi); out.print(' ');
//				out.print(type); out.print(' ');
//				out.print(alphabet.lookupObject(type)); out.print(' ');
//				out.print(topic); out.println();
//				*/
//			}
//
//			out.print(output);
//		}
//	}
	
	public double perplexity()
	{

		double pp = 0.0;
		int[] topicCounts = new int[numTopics];
		int[] docTopics;
		
		double[] docTheta = new double[numTopics];
		
		double[][] phi = new double[numTopics][];
		for(int topic = 0; topic < numTopics; topic ++)
		{
			phi[topic] = new double[numTypes];
		}
		
		
		for (int type=0; type < numTypes; type++) 
		{
			topicCounts = typeTopicCounts[type];
			
			for(int topic = 0; topic < numTopics; topic ++)
			{
				phi[topic][type] = beta[topic][type] / betaSum[topic];
			}

			int index = 0;
			while (index < topicCounts.length &&
				   topicCounts[index] > 0) {
				int topic = topicCounts[index] & topicMask;
				int count = topicCounts[index] >> topicBits;
				
				phi[topic][type] =  (beta[topic][type] + count) / ( betaSum[topic] + this.tokensPerTopic[topic]); 

				index++;
			}
		}
		int[] docTopicCounts = new int[numTopics];

		
		for (int doc=0; doc < data.size(); doc++) 
		{
			
			Arrays.fill(docTopicCounts, 0);
			FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).instance.getData();

			int docLength = tokenSequence.getLength();
			LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;

			docTopics = topicSequence.getFeatures();

			for (int token=0; token < docTopics.length; token++) {
				docTopicCounts[ docTopics[token] ]++;
			}

			for (int topic=0; topic < numTopics; topic++) {
				
				docTheta[topic] = (alpha[doc][topic] + docTopicCounts[topic]) / (alphaSum[doc] + docLength);
				
			}
			for (int position = 0; position < docLength; position++) 
			{
				double wordProb = 0;
				int type = tokenSequence.getIndexAtPosition(position);
				for(int topic = 0; topic < numTopics; topic ++)
				{
					wordProb += docTheta[topic] * phi[topic][type];
				}
				pp += Math.log(wordProb);
				
			}

		}
		
		return Math.exp(- pp / totalTokens) ;
	}
	
	public double modelLogLikelihood() {
		double logLikelihood = 0.0;
		int nonZeroTopics;

		// The likelihood of the model is a combination of a 
		// Dirichlet-multinomial for the words in each topic
		// and a Dirichlet-multinomial for the topics in each
		// document.

		// The likelihood function of a dirichlet multinomial is
		//	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
		//	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )

		// So the log likelihood is 
		//	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) + 
		//	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]

		// Do the documents first

		int[] topicCounts = new int[numTopics];
		double[][] topicLogGammas = new double[data.size()][numTopics];
		int[] docTopics;

	
		for (int doc=0; doc < data.size(); doc++) {
			LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;

			docTopics = topicSequence.getFeatures();

			for (int token=0; token < docTopics.length; token++) {
				topicCounts[ docTopics[token] ]++;
			}

			for (int topic=0; topic < numTopics; topic++) {
				if (topicCounts[topic] > 0) {
					double docTopicLogGamma = Dirichlet.logGammaStirling( alpha[doc][topic] );
					logLikelihood += (Dirichlet.logGammaStirling(alpha[doc][topic] + topicCounts[topic]) -
							docTopicLogGamma);
				}
			}

			// subtract the (count + parameter) sum term
			logLikelihood -= Dirichlet.logGammaStirling(alphaSum[doc] + docTopics.length);

			Arrays.fill(topicCounts, 0);
		}

		// add the parameter sum term
		
		for (int doc=0; doc < data.size(); doc++) {

			logLikelihood += Dirichlet.logGammaStirling(alphaSum[doc]);
		}
		// And the topics

		// Count the number of type-topic pairs that are not just (logGamma(beta) - logGamma(beta))

		for (int type=0; type < numTypes; type++) {
			// reuse this array as a pointer

			topicCounts = typeTopicCounts[type];

			int index = 0;
			while (index < topicCounts.length &&
				   topicCounts[index] > 0) {
				int topic = topicCounts[index] & topicMask;
				int count = topicCounts[index] >> topicBits;
				
				logLikelihood +=  Dirichlet.logGammaStirling(beta[topic][type] + count);

				if (Double.isNaN(logLikelihood)) {
					logger.warning("NaN in log likelihood calculation");
					return 0;
				}
				else if (Double.isInfinite(logLikelihood)) {
					logger.warning("infinite log likelihood");
					logger.warning(Double.toString(beta[topic][type]));
					logger.warning(Double.toString(count));
					return 0;
				}
				
				logLikelihood -=
						Dirichlet.logGammaStirling(beta[topic][type]);
				

				index++;
			}
		}
	
		for (int topic=0; topic < numTopics; topic++) {
			logLikelihood -= 
				Dirichlet.logGammaStirling( (betaSum[topic]) +
											tokensPerTopic[ topic ] );

			if (Double.isNaN(logLikelihood)) {
				logger.info("NaN after topic " + topic + " " + tokensPerTopic[ topic ]);
				return 0;
			}
			else if (Double.isInfinite(logLikelihood)) {
				logger.info("Infinite value after topic " + topic + " " + tokensPerTopic[ topic ]);
				return 0;
			}
			logLikelihood += 
					Dirichlet.logGammaStirling(betaSum[topic]);

		}
	


		if (Double.isNaN(logLikelihood)) {
			logger.info("at the end");
		}
		else if (Double.isInfinite(logLikelihood)) {
			logger.info("Infinite value beta " + beta + " * " + numTypes);
			return 0;
		}

		return logLikelihood;
	}





	
}

