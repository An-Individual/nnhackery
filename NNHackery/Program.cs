using NNHackery.Components;
using NNHackery.LinearAlgebra;
using NNHackery.MNIST;
using NNHackery.Trainers;

namespace NNHackery
{
    internal class Program
    {
        static void Main(string[] args)
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

            int totalEpochs = 1000;
            int batchSize = 10;
            double learningRate = 3;

            MNISTTestImage[] testBatch = new MNISTTestImage[batchSize];

            for(int epoch = 1; epoch <= totalEpochs; epoch++)
            {
                Console.WriteLine($"Running Epoch {epoch}:");
                Console.WriteLine($"    Shuffling mini-batches");
                Random.Shared.Shuffle(trainingData);

                Console.Write("    Executing mini-batch 0");
                int batchCount = 0;
                foreach (MNISTTestImage[] batch in trainingData.Chunk(batchSize))
                {
                    for(int i = 0; i < batchCount.ToString().Length; i++)
                    {
                        Console.Write("\b");
                    }
                    batchCount++;
                    Console.Write(batchCount);

                    Vector[] inputs = batch.Select(i => i.FlattenedImage).ToArray();
                    Vector[] expected = batch.Select(i => i.LabelVector).ToArray();

                    QuadraticTrainer.RunGradientDescent(network, inputs, expected, learningRate);
                }

                Console.WriteLine();
                Console.WriteLine("    Testing network...");

                int passes = 0;
                foreach(MNISTTestImage testImage in testData)
                {
                    Vector output = network.ApplyNetwork(testImage.FlattenedImage);

                    if(testImage.Label == GetHighestVectorIndex(output))
                    {
                        passes++;
                    }
                }

                Console.WriteLine($"    Passes: {passes} / {testData.Length}");
            }
        }

        private static int GetHighestVectorIndex(Vector vector)
        {
            int highestIndex = 0;
            double highestValue = vector[0];
            for(int i = 1; i < vector.Size; i++)
            {
                if (vector[i] > highestValue)
                {
                    highestIndex = i;
                }
            }

            return highestIndex;
        }

        private static void RandomizeNetwork(Network network)
        {
            Parallel.ForEach(network.Layers, layer =>
            {
                Parallel.For(0, layer.Weights.Width, x =>
                {
                    Parallel.For(0, layer.Weights.Height, y =>
                    {
                        layer.Weights[x, y] = Random.Shared.NextDouble();
                    });
                });

                Parallel.For(0, layer.Biases.Size, i =>
                {
                    layer.Biases[i] = Random.Shared.NextDouble();
                });
            });
        }
    }
}
