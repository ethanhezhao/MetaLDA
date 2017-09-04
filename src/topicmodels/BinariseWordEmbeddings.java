package topicmodels;

import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.util.CommandOption;
import com.google.common.base.Charsets;
import com.google.common.io.Files;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhaohe on 28/08/2017.
 */
public class BinariseWordEmbeddings {



    public static ArrayList<int[]> binariseWordEmbeddings(ArrayList<double[]> we)
    {
        ArrayList<int[]> bwe = new ArrayList<int[]>();

        int L = we.get(0).length;

        int V = we.size();

        for(int v = 0; v < V; v++)
        {

            int[] bwe_v = new int[L * 2];
            double posMean = 0.0;

            double negMean = 0.0;

            int posCount = 0;

            int negCount = 0;

            for (int l = 0; l < L; l++)
            {
                if (we.get(v)[l] > 0)
                {
                    posMean += we.get(v)[l];
                    posCount ++;
                }
                else if(we.get(v)[l] < 0)
                {
                    negMean += we.get(v)[l];
                    negCount ++;
                }
            }

            posMean /= posCount;

            negMean /= negCount;

            for (int l = 0; l < L; l++) {
                if (we.get(v)[l] > posMean)
                {
                    bwe_v[2*l] = 1;
                }
                else if(we.get(v)[l] < negMean)
                {
                    bwe_v[2*l + 1] = 1;
                }
            }
            bwe.add(bwe_v);

        }

        return bwe;
    }


    static CommandOption.String trainDocInputFile =
            new cc.mallet.util.CommandOption.String(BinariseWordEmbeddings.class, "train-docs", "FILENAME", true, null,
                    "The filename from which to read the list of training instances.  Use - for stdin.  " +
                            "The instances must be FeatureSequence, not FeatureVector", null);

    static CommandOption.String testDocInputFile =
            new cc.mallet.util.CommandOption.String(BinariseWordEmbeddings.class, "test-docs", "FILENAME", true, null,
                    "The filename from which to read the list of test instances.  Use - for stdin.  " +
                            "The instances must be FeatureSequence, not FeatureVector", null);


    static CommandOption.String wordOutputFile =
            new cc.mallet.util.CommandOption.String(BinariseWordEmbeddings.class, "output", "FILENAME", true, null,
                    "The filename to which to write the binarised word embeddings", null);


    static CommandOption.String wordInputFile =
            new cc.mallet.util.CommandOption.String(BinariseWordEmbeddings.class, "input", "FILENAME", true, null,
                    "The filename from which to read the word embeddings", null);

    public static void main(String[] args) throws IOException {
        CommandOption.setSummary (BinariseWordEmbeddings.class,
                "Binarise word embeddings");
        CommandOption.process (BinariseWordEmbeddings.class, args);



        Alphabet trainAlphabet = null;
        if (trainDocInputFile.value != null)
            trainAlphabet = InstanceList.load(new File(trainDocInputFile.value)).getAlphabet();


        Alphabet testAlphabet = null;
        if (testDocInputFile.value != null)
            testAlphabet = InstanceList.load(new File(testDocInputFile.value)).getAlphabet();


        File wordFeatureFile = new File(wordInputFile.value);

        List<String> lines;
        try {
            lines = Files.readLines(wordFeatureFile, Charsets.UTF_8);
        } catch (IOException e) {

            lines = null;
        }




        ArrayList<String> wordList = new ArrayList<String>();

        ArrayList<double[]> we = new ArrayList<double[]>();

        if (lines != null) {




            for (String line : lines) {


                String[] ls = line.trim().split(" ");

                boolean isInterested = false;
                if(trainAlphabet != null)
                {
                    int idx = trainAlphabet.lookupIndex(ls[0],false);

                    if(idx > -1 && idx < trainAlphabet.size())
                    {
                        isInterested = true;
                    }
                }

                if(testAlphabet != null)
                {
                    int idx = testAlphabet.lookupIndex(ls[0],false);

                    if(idx > -1 && idx < testAlphabet.size())
                    {
                        isInterested = true;
                    }
                }

                if(isInterested) {

                    double[] we_v = new double[ls.length - 1];

                    for (int i = 1; i < ls.length; i++) {
                        we_v[i - 1] = Double.parseDouble(ls[i]);

                    }
                    we.add(we_v);
                    wordList.add(ls[0]);

                }




            }

            ArrayList<int[]> bwe = binariseWordEmbeddings(we);


            BufferedWriter outputWriter = null;

            outputWriter = new BufferedWriter(new FileWriter(wordOutputFile.value));



            for(int v = 0; v < bwe.size(); v ++)
            {
                outputWriter.write(wordList.get(v) + "\t");

                int[] bwe_v = bwe.get(v);
                for(int l = 0; l < bwe_v.length; l ++)
                {
                    if(bwe_v[l] > 0)
                        outputWriter.write(Integer.toString(l) + " ");
                }
                outputWriter.newLine();
            }

            outputWriter.flush();
            outputWriter.close();


        }



    }



}
