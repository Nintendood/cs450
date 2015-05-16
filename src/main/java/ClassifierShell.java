import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.Id3;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.Filter;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class ClassifierShell {
    public static void main(String[] args) throws Exception {
            DataSource source = new DataSource("house-votes-84.csv");
            Instances dataSetPre = source.getDataSet();

            dataSetPre.setClassIndex(dataSetPre.numAttributes() - 1);
            //dataSetPre.setClassIndex(0);

            Standardize stand = new Standardize();
            stand.setInputFormat(dataSetPre);

            Discretize discretize = new Discretize();
            discretize.setInputFormat(dataSetPre);

            Instances dataSet = dataSetPre;

            dataSet = Filter.useFilter(dataSet, discretize);
            dataSet = Filter.useFilter(dataSet, stand);

            dataSet.randomize(new Random(9001));

            //My home-grown classifier
            Classifier classifier = new ID3();

            //Weka's classifier that doesn't take nominal data
            //Id3 classifier = new Id3();
            Evaluation eval = new Evaluation(dataSet);
            final int folds = 10;
            for (int n = 0; n < folds; n++) {
                Instances train = dataSet.trainCV(folds, n);
                Instances test = dataSet.testCV(folds, n);

                Classifier clsCopy = Classifier.makeCopy(classifier);
                clsCopy.buildClassifier(train);
                eval.evaluateModel(clsCopy, test);
            }

            System.out.println(eval.toSummaryString("\n" + folds + " Fold Cross Validation\n==============================================="
                    + "===================", false));


        }
}
