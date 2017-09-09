package topicmodels;

import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.InstanceList;
import cc.mallet.util.CommandOption;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by zhaohe on 7/09/2017.
 */
public class MalletLDATrain {
    static CommandOption.String docInputFile =
            new cc.mallet.util.CommandOption.String(MalletLDATrain.class, "train-docs", "FILENAME", true, null,
                    "The filename from which to read the list of training instances.  Use - for stdin.  " +
                            "The instances must be FeatureSequence, not FeatureVector", null);




    static CommandOption.String saveFolderName =
            new cc.mallet.util.CommandOption.String(MalletLDATrain.class, "save-folder", "FILENAME", true, null,
                    "the folder that saves the statistics", null);


    // Model parameters

    static CommandOption.Integer numIterationsOption =
            new CommandOption.Integer(MalletLDATrain.class, "num-iterations", "INTEGER", true, 2000,
                    "The number of iterations of Gibbs sampling. Default is 2000.", null);



    static CommandOption.Integer randomSeedOption =
            new CommandOption.Integer(MalletLDATrain.class, "random-seed", "INTEGER", true, -1,
                    "The random seed for the Gibbs sampler.  Default is -1, which will use the clock.", null);

    // Hyperparameters



    static CommandOption.Boolean isSymmetricAlpha =
            new CommandOption.Boolean(MalletLDATrain.class, "sym-alpha", "BOOLEAN", true, true,
                    "whether alpha is sampled symmetrically. Default is true", null);



    static CommandOption.Integer sampleIntervalOption =
            new CommandOption.Integer(MalletLDATrain.class, "sample-interval", "INTEGER", true, 1,
                    "The number of iterations between sampling alpha and beta. Default is 1", null);

    static CommandOption.Integer burninPeriodOption =
            new CommandOption.Integer(MalletLDATrain.class, "burn-in-period", "INTEGER", true, 10,
                    "The number of iterations in burn-in period before sampling alpha and/or beta. Default is 10", null);




    static CommandOption.Integer numTopicsOption =
            new CommandOption.Integer(MalletLDATrain.class, "num-topics", "INTEGER", true, 50,
                    "The number of topics. Default is 50", null);


    static CommandOption.Integer numThreadsOption =
            new CommandOption.Integer(MalletLDATrain.class, "num-threads", "INTEGER", true, 1,
                    "The number of threads. Default is 1", null);

    static CommandOption.Double initAlpha =
            new CommandOption.Double(MalletLDATrain.class, "initial-alpha", "DOUBLE", true, 0.1,
                    "The initial value of alpha. Default is 0.1", null);

    static CommandOption.Double initBeta =
            new CommandOption.Double(MalletLDATrain.class, "initial-beta", "DOUBLE", true, 0.01,
                    "The initial value of beta. Default is 0.01", null);




    public static void main(String[] args) throws IOException {


        CommandOption.setSummary(MalletLDATrain.class,
                "MetaLDA, a topic model that incorporates meta information");
        CommandOption.process(MalletLDATrain.class, args);


        if (docInputFile.value == null) {
            CommandOption.getList(MalletLDATrain.class).printUsage(false);
            System.exit(-1);
        }

        if (saveFolderName.value == null) {
            CommandOption.getList(MetaLDATrain.class).printUsage(false);
            System.exit(-1);
        }


        ParallelTopicModel lda = null;

        InstanceList training = InstanceList.load(new File(docInputFile.value));
        lda = new ParallelTopicModel(numTopicsOption.value, initAlpha.value * numTopicsOption.value, initBeta.value);


        lda.setOptimizeInterval(sampleIntervalOption.value);

        lda.setBurninPeriod(burninPeriodOption.value);

        lda.setNumIterations(numIterationsOption.value);

        lda.setNumThreads(numThreadsOption.value);

        lda.setRandomSeed(randomSeedOption.value);

        lda.setSymmetricAlpha(isSymmetricAlpha.value);


        lda.addInstances(training);

        //do not display topics while training
        lda.setTopicDisplay(0, 0);


        try {
            lda.estimate();
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (saveFolderName.value != null && lda != null) {


            File saveDir = new File(saveFolderName.value);

            if (!saveDir.exists()) {

                saveDir.mkdir();

            }


            saveFiles(lda, saveFolderName.value);


        }
    }


    public static void saveFiles(ParallelTopicModel lda, String folderName) throws IOException {


        int numDocs = lda.getData().size();



        File matSaveFile = new File(folderName + "/train_stats.mat");


        ArrayList saveList = new ArrayList();

        double[][] alpha = new double[numDocs][];

        for(int i = 0; i < numDocs; i ++) {
            alpha[i] = new double[lda.getNumTopics()];
            System.arraycopy(lda.alpha,0,alpha[i],0,lda.alpha.length);
        }

        MLDouble alphaMat = null;


        alphaMat = new MLDouble("alpha", alpha);


        saveList.add( alphaMat );


        double[][] beta = new double[lda.getNumTopics()][];

        for(int k = 0; k < lda.getNumTopics(); k ++) {
            beta[k] = new double[lda.getAlphabet().size()];
            Arrays.fill(beta[k],lda.beta);
        }


        MLDouble betaMat = null;

        betaMat = new MLDouble("beta", beta);

        saveList.add(betaMat);

        MLDouble docTopicCounts = new MLDouble("doc_topic", lda.getDocumentTopics(false, false));

        saveList.add(docTopicCounts);

        MLDouble topicTypeCounts = new MLDouble("topic_type", lda.getTopicWords(false,false));

        saveList.add(topicTypeCounts);




        int sampleAlphaMethod = 0;

        int sampleBetaMethod = 0;

        if (lda.optimizeInterval > 0)
        {
            sampleBetaMethod = 4;

            if (lda.usingSymmetricAlpha)
            {
                sampleAlphaMethod = 5;
            }
            else
            {
                sampleAlphaMethod = 4;
            }
        }


        saveList.add(new MLChar("sampleAlphaMethod", Integer.toString(sampleAlphaMethod)));
        saveList.add(new MLChar("sampleBetaMethod", Integer.toString(sampleBetaMethod)));

        saveList.add(new MLChar("numTopics", Integer.toString(lda.getNumTopics())));

        saveList.add(new MLChar("numIterations", Integer.toString(lda.numIterations)));

        saveList.add(new MLChar("burninPeriod", Integer.toString(lda.burninPeriod)));

        saveList.add(new MLChar("optimizeAlphaInterval", Integer.toString(lda.optimizeInterval)));
        saveList.add(new MLChar("optimizeBetaInterval", Integer.toString(lda.optimizeInterval)));





        new MatFileWriter(matSaveFile, saveList);





        File trainAlphabetFile = new File(folderName + "/train_alphabet.txt");

        PrintStream trainAlphabetOut = new PrintStream(trainAlphabetFile);

        trainAlphabetOut.println(lda.data.get(0).instance.getDataAlphabet().toString());

        trainAlphabetOut.close();


        File trainTargetAlphabetFile = new File(folderName + "/train_target_alphabet.txt");

        PrintStream trainTargetAlphabetOut = new PrintStream(trainTargetAlphabetFile);

        trainTargetAlphabetOut.println(lda.data.get(0).instance.getTargetAlphabet().toString());

        trainTargetAlphabetOut.close();



        lda.printTopWords(new File(folderName + "/top_words.txt"),50, false);



    }


}
