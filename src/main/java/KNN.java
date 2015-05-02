import com.sun.corba.se.impl.encoding.OSFCodeSetRegistry;
import sun.beans.editors.DoubleEditor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by Adam Harris on 4/29/2015. Wut Wut.
 */
public class KNN extends Classifier {

    Instances saved;
    final int k;

    public KNN(int k)
    {
        this.k = k;
    }

    public static double getClassification(List<Instance> instances)
    {
        int index = instances.get(0).classIndex();
        HashMap<Double,Integer> counts = new HashMap<Double, Integer>();

        double total = 0;
        for (Instance instance : instances) {
            double val = instance.value(index);
            if (!counts.containsKey(val)) {
                counts.put(val, 1);
            } else {
                counts.put(val, (counts.get(val)) + 1);
            }
        }

        int maxCount = 0;
        double maxValue = 0;

        for (Entry<Double, Integer> entry : counts.entrySet())
        {
            if (entry.getValue() > maxCount)
            {
                maxCount = entry.getValue();
                maxValue = entry.getKey();
            }
        }

        return maxValue;
    }

    private static double distance(Instance one, Instance two)
    {
        double total = 0;
        int totalAttributes = one.numAttributes() - 1;
        for (int i = 0; i < totalAttributes; i++)
        {
            if (one.classIndex() == i)
                continue;

            double difference = 0;

            if (one.attribute(i).isNumeric())
            {
                difference = Math.abs(one.value(i) - two.value(i));
            }
            else
            {
                if (!one.stringValue(i).equals(two.stringValue(i)))
                {
                    difference = 1;
                }
            }

            total += Math.pow(difference, totalAttributes);
        }
        return Math.pow(total, 1.0/totalAttributes);
    }

    @Override
    public void buildClassifier(Instances instances)// throws Exception
    {
        saved = new Instances(instances);
    }

    @Override
    public double classifyInstance(Instance instance)// throws Exception
    {
        HashMap<Instance, Double> map = new HashMap<Instance, Double>();

        for (int i = 0; i < saved.numInstances(); i++)
        {
            Instance tmp = saved.instance(i);
            map.put(tmp, distance(tmp, instance));
        }

        ArrayList<Entry<Instance, Double>> sorted = new ArrayList<Entry<Instance, Double>>(map.entrySet());
        Collections.sort(sorted, new Comparator<Entry<Instance, Double>>() {
            @Override
            public int compare(Entry<Instance, Double> e1, Entry<Instance, Double> e2) {
                return e1.getValue().compareTo(e2.getValue());
            }
        });

        List<Instance> kNearest = new ArrayList<Instance>();

        for (Entry<Instance, Double> inst : sorted)
        {
            kNearest.add(inst.getKey());

            if (kNearest.size() >= k)
            {
                break;
            }
        }
        return getClassification(kNearest);
    }
}
