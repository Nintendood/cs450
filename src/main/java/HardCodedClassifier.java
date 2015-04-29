import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class HardCodedClassifier extends Classifier {
    @Override
    public void buildClassifier(Instances instances) throws Exception {

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }
}
