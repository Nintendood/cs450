import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class NNC extends Classifier {

    Network network;
    int layers = 3;
    int iterations = 50;
    double learnFactor = 0.3;

    public NNC(int numLayers, int iterations, double learnFactor){
        this.layers = numLayers;
        this.iterations = iterations;
        this.learnFactor = learnFactor;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        int inputCount = instances.numAttributes() - 1;

        List<Integer> nodesPerLayer = new ArrayList<Integer>();

        for (int i = 0; i < layers - 1; i++)
        {
            nodesPerLayer.add(inputCount);
        }

        nodesPerLayer.add(instances.numDistinctValues(instances.classIndex()));

        network = new Network(inputCount, nodesPerLayer);

        ArrayList<Double> errorsPerIteration = new ArrayList<Double>();
        for (int j = 0; j < iterations; j++)
        {
            for (int k = 0; k < instances.numInstances(); k++)
            {
                Instance instance = instances.instance(k);

                List<Double> input = new ArrayList<Double>();
                for (int i = 0; i < instance.numAttributes(); i++)
                {
                    if (Double.isNaN(instance.value(i)) && i != instance.classIndex())
                    {
                        input.add(0.0);
                    }
                    else if (i != instance.classIndex())
                    {
                        input.add(instance.value(i));
                    }
                }
                errorsPerIteration.add(network.train(input, instance.value(instance.classIndex()), learnFactor));
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        List<Double> input = new ArrayList<Double>();

        for (int i = 0; i < instance.numAttributes(); i++)
        {
            if (Double.isNaN(instance.value(i)) && i != instance.classIndex())
            {
                input.add(0.0);
            }
            else if (i != instance.classIndex())
            {
                input.add(instance.value(i));
            }
        }

        List<Double> outputs = network.getOutputs(input);
        double largeValue = -1.0;
        int index = 0;
        for (int i = 0; i < outputs.size(); i++)
        {
            double temp = outputs.get(i);
            if (temp > largeValue)
            {
                largeValue = temp;
                index = i;
            }
        }

        return index;
    }
}

class Neuron {
    List<Double> weights = new ArrayList<Double>();
    static Random R = new Random(42);

    //constructor
    public Neuron(int numInputs)
    {
        Double oneOver = (1.0/ Math.sqrt(numInputs));
        //Random R = new Random(42);
        for (int i = 0; i < numInputs; i++)
        {
            weights.add(R.nextDouble() * 2.0 * oneOver - oneOver);
        }
    }

    public double produceOutput (List<Double> inputs)
    {
        if (inputs.size() != weights.size())
        {
            throw new UnsupportedOperationException("Wrong number of inputs. We expect "
                    + weights.size() + " and we got " + inputs.size());
        }

        double sum = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            sum += (weights.get(i) * inputs.get(i));
        }

        return 1.0 / (1.0 + Math.exp(-sum));
    }
}

class Layer{
    List<Neuron> neurons =new  ArrayList<Neuron>();

    public Layer(int numNeurons, int numInputs)
    {
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.add(new Neuron(numInputs));
        }
    }

    public List<Double> produceOutputs(List<Double> inputs)
    {
        List<Double> outputs = new ArrayList<Double>();
        for (Neuron neuron: neurons)
        {
            outputs.add(neuron.produceOutput(inputs));
        }
        return outputs;
    }
}

class Network{
    List<Layer> layers = new ArrayList<Layer>();

    public Network(int numInputs, List<Integer> neuronsPerLayer)
    {
        if (neuronsPerLayer.isEmpty())
        {
            throw new UnsupportedOperationException("neuronsPerLayer is empty.");
        }

        layers.add(new Layer(neuronsPerLayer.get(0), numInputs + 1));

        for (int i = 1; i < neuronsPerLayer.size(); i++)
        {
            layers.add(new Layer(neuronsPerLayer.get(i), neuronsPerLayer.get(i - 1) + 1));
        }
    }

    public List<Double> getOutputs(List<Double> inputs)
    {
        List<Double> outputs = new ArrayList<Double>(inputs);

        for (Layer layer: layers)
        {
            //add bias
            outputs.add(1.0);
            outputs = layer.produceOutputs(outputs);
        }

        return outputs;
    }

    public double train(List<Double> inputs, double classification, double learnValue)
    {
        ArrayList<List<Double>> all = new ArrayList<List<Double>>();
        List<Double> outputs = new ArrayList<Double>(inputs);

        for (Layer layer: layers)
        {
            outputs.add(1.0);
            outputs = layer.produceOutputs(outputs);
            all.add(outputs);
            for (Double d: outputs)
            {
                //blank
            }
        }

        ArrayList<ArrayList<Double>> allErrors = new ArrayList<ArrayList<Double>>();

        ArrayList<Double> error = new ArrayList<Double>();
        List<Double> currentOutputs = all.get(all.size() - 1);
        Layer current = layers.get(layers.size() - 1);
        for (int i = 0; i < current.neurons.size(); i++)
        {
            double expected = (classification == i ? 1 : 0);
            error.add(currentOutputs.get(i) * (1 - currentOutputs.get(i)) * (currentOutputs.get(i) - expected));
        }

        allErrors.add(error);


        for (int i = layers.size() - 2; i >= 0; i--)
        {
            current = layers.get(i);
            error = new ArrayList<Double>();
            outputs = all.get(i);
            ArrayList<Double> followingError = allErrors.get(0);
            for (int j = 0; j < current.neurons.size(); j++)
            {
                double sumError = 0;

                Layer nextLayer = layers.get(i + 1);
                for (int k = 0; k < followingError.size(); k++)
                {
                    sumError += followingError.get(k) * nextLayer.neurons.get(k).weights.get(j);
                }

                double errorVal = outputs.get(j) * (1 - outputs.get(j)) * sumError;
                error.add(errorVal);
            }

            allErrors.add(0, error);
        }

        inputs.add(1.0);
        all.add(0,inputs);
        for (int i = 0; i < layers.size(); i++)
        {
            current = layers.get(i);
            for (int j = 0; j < current.neurons.size(); j++)
            {
                Neuron neuron = current.neurons.get(j);
                for (int k = 0; k < neuron.weights.size(); k++)
                {
                    double newWeight = neuron.weights.get(k) - all.get(i).get(k) * allErrors.get(i).get(j) * learnValue;
                    neuron.weights.set(k, newWeight);
                }
            }
        }


        double totalError = 0;
        for (List<Double> l : allErrors)
        {
            for (Double d : l)
            {
                totalError += Math.abs(d);
            }
        }

        return totalError;
    }
}