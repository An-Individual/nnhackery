using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NNHackery.Components;
using NNHackery.MNIST;
using NNHackery.Trainers;
using System.Collections.Concurrent;

namespace NNHackery
{
    internal class Program
    {
        static void Main(string[] args)
        {
            RunQuadraticMNISTTest(30, 10, 3);
        }

        private static void RunQuadraticMNISTTest(int epochs, int batchSize, double learningRate)
        {
            Console.WriteLine("Initializing network...");
            Network network = new Network(28 * 28, 30, 10);

            Console.WriteLine("Randomizing network...");
            RandomizeNetwork(network);

            Console.WriteLine("Loading MNIST data...");
            MNISTTestImage[] trainingData = MNISTReader.ReadTrainingImages();

            Console.WriteLine("Selecting test cases...");
            Random.Shared.Shuffle(trainingData);
            MNISTTestImage[] testData = trainingData.Chunk(10000).Last();
            trainingData = trainingData.SkipLast(10000).ToArray();

            MNISTTestImage[] testBatch = new MNISTTestImage[batchSize];
            ITrainer trainer = new QuadraticTrainer();

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine($"Running Epoch {epoch}:");
                Console.WriteLine($"    Shuffling mini-batches");
                Random.Shared.Shuffle(trainingData);

                Console.Write("    Executing mini-batch 0");
                int batchCount = 0;
                foreach (MNISTTestImage[] batch in trainingData.Chunk(batchSize))
                {
                    for (int i = 0; i < batchCount.ToString().Length; i++)
                    {
                        Console.Write("\b");
                    }
                    batchCount++;
                    Console.Write(batchCount);

                    Vector<double>[] inputs = batch.Select(i => i.FlattenedImage).ToArray();
                    Vector<double>[] expected = batch.Select(i => i.LabelVector).ToArray();

                    trainer.RunGradientDescent(network, inputs, expected, learningRate);
                }

                Console.WriteLine();
                Console.WriteLine("    Testing network...");

                int passes = 0;
                double totalCost = 0;
                foreach (MNISTTestImage testImage in testData)
                {
                    Vector<double> output = network.ApplyNetwork(testImage.FlattenedImage);

                    if (testImage.Label == GetHighestVectorIndex(output))
                    {
                        passes++;
                    }

                    totalCost += (output - testImage.LabelVector).Select(v => Math.Pow(v, 2)).Aggregate((c, v) => c + v);
                }

                Console.WriteLine($"    Passes: {passes} / {testData.Length}");
                Console.WriteLine($"    Average Cost: {totalCost / testData.Length}");
            }
        }

        private static int GetHighestVectorIndex(Vector<double> vector)
        {
            int highestIndex = 0;
            double highestValue = vector[0];
            for(int i = 1; i < vector.Count; i++)
            {
                if (vector[i] > highestValue)
                {
                    highestValue = vector[i];
                    highestIndex = i;
                }
            }

            return highestIndex;
        }

        private static void RandomizeNetwork(Network network)
        {
            int totalSamples = network.Layers
                .Select(l => l.Weights.ColumnCount * l.Weights.RowCount + l.Biases.Count)
                .Aggregate((c,v) => c + v);
            double[] samplesRaw = new double[totalSamples];

            Normal distribution = new Normal();
            distribution.Samples(samplesRaw);

            ConcurrentQueue<double> samples = new ConcurrentQueue<double>(samplesRaw);

            Parallel.ForEach(network.Layers, layer =>
            {
                Parallel.For(0, layer.Weights.ColumnCount, col =>
                {
                    Parallel.For(0, layer.Weights.RowCount, row =>
                    {
                        if(!samples.TryDequeue(out double value))
                        {
                            throw new Exception("Out of samples.");
                        }

                        layer.Weights[row, col] = value;
                    });
                });

                Parallel.For(0, layer.Biases.Count, i =>
                {
                    if (!samples.TryDequeue(out double value))
                    {
                        throw new Exception("Out of samples.");
                    }

                    layer.Biases[i] = value;
                });
            });
        }
    }
}
