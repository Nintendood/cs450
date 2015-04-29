import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class ClassifierShell {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("irisdata.csv");
        Instances dataSet = source.getDataSet();

        dataSet.setClassIndex(dataSet.numAttributes() - 1);

        dataSet.randomize(new Random(1));

        int trainSize = (int) Math.round(dataSet.numInstances() * .7);
        int testSize = dataSet.numInstances() - trainSize;
        Instances train = new Instances(dataSet, 0, trainSize);
        Instances test = new Instances(dataSet, trainSize, testSize);

        HardCodedClassifier hcc = new HardCodedClassifier();
        hcc.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(hcc, test);
        System.out.println(eval.toSummaryString("\nResults\n", true));


    }
}
