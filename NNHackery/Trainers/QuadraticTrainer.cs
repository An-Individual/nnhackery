using MathNet.Numerics.LinearAlgebra;
using NNHackery.Components;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.XPath;

namespace NNHackery.Trainers
{
    public class QuadraticTrainer : ITrainer
    {
        public class IterationState
        {
            public Network Network { get; set; }

            public Vector<double> Input { get; set; }

            public Vector<double> Expected { get; set; }

            public Vector<double>[] Vectors { get; set; }

            public Vector<double>[] Activations { get; set; }

            public Vector<double>[] Errors { get; set; }
        }

        public class LayerGradient
        {
            public Matrix<double> WeightGrade { get; set; }

            public Vector<double> BiaseGrade { get; set; }

            public void ApplyToLayer(Layer layer, double learningRate, int batchSize)
            {
                layer.Weights -= ((learningRate / batchSize) * WeightGrade);
                layer.Biases -= ((learningRate / batchSize) * BiaseGrade);
            }
        }

        public class TrainingRun
        {
            public TrainingRun(Network network)
            {
                Gradients = new LayerGradient[network.Layers.Length];
            }

            public LayerGradient[] Gradients { get; private set; }

            private readonly object _lock = new object();

            public void AddCase(LayerGradient[] testCase)
            {
                ArgumentNullException.ThrowIfNull(testCase);

                if (testCase.Length != Gradients.Length)
                {
                    throw new Exception("Test case has the wrong number of gradients.");
                }

                lock (_lock)
                {
                    Parallel.For(0, Gradients.Length, i =>
                    {
                        if (Gradients[i] == null)
                        {
                            Gradients[i] = testCase[i];
                        }
                        else
                        {
                            Gradients[i].WeightGrade += testCase[i].WeightGrade;
                            Gradients[i].BiaseGrade += testCase[i].BiaseGrade;
                        }
                    });
                }
            }
        }

        public void RunGradientDescent(Network network, Vector<double>[] inputs, Vector<double>[] expectedOutputs, double learningRate)
        {
            ArgumentNullException.ThrowIfNull(network);
            ArgumentNullException.ThrowIfNull(inputs);
            ArgumentNullException.ThrowIfNull(expectedOutputs);

            if (inputs.Length != expectedOutputs.Length)
            {
                throw new Exception("Inputs and expected outputs must be same length.");
            }

            TrainingRun run = new TrainingRun(network);

            Parallel.For(0, inputs.Length, i =>
            {
                LayerGradient[] gradients = CalculateGradients(network, inputs[i], expectedOutputs[i]);
                run.AddCase(gradients);
            });

            if(run.Gradients.Length != network.Layers.Length)
            {
                throw new Exception("Run produced the wrong number of gradients.");
            }

            Parallel.For(0, network.Layers.Length, i =>
            {
                run.Gradients[i].ApplyToLayer(network.Layers[i], learningRate, inputs.Length);
            });
        }

        public static LayerGradient[] CalculateGradients(Network network, Vector<double> input, Vector<double> expected)
        {
            IterationState state = new IterationState();
            state.Network = network;
            state.Input = input;
            state.Expected = expected;

            CalculateVectorsAndActivations(state);
            CalculateErrors(state);

            LayerGradient[] result = new LayerGradient[state.Network.Layers.Length];

            Parallel.For(0, result.Length, i =>
            {
                Vector<double> activation;
                if(i == 0)
                {
                    activation = state.Input;
                }
                else
                {
                    activation = state.Activations[i - 1];
                }

                result[i] = CalculateLayerGradients(activation, state.Errors[i]);
            });

            return result;
        }

        public static void CalculateVectorsAndActivations(IterationState state)
        {
            state.Vectors = new Vector<double>[state.Network.Layers.Length];
            state.Activations = new Vector<double>[state.Network.Layers.Length];

            Vector<double> currentVector = state.Input;

            for (int i = 0; i < state.Network.Layers.Length; i++)
            {
                currentVector = state.Network.Layers[i].ApplyToVector(currentVector);

                state.Vectors[i] = currentVector;
                state.Activations[i] = currentVector.Clone();
                state.Activations[i].MapInplace(state.Network.ActivationFunction);
                currentVector = state.Activations[i];
            }
        }

        public static void CalculateErrors(IterationState state)
        {
            state.Errors = new Vector<double>[state.Network.Layers.Length];

            int lastIndex = state.Network.Layers.Length - 1;

            for (int i = lastIndex; i >= 0; i--)
            {
                Vector<double> prime = state.Vectors[i];
                prime.MapInplace(state.Network.ActivationDerivative);

                Vector<double> deltaC;
                if(i == lastIndex)
                {
                    deltaC = state.Activations[lastIndex] - state.Expected;
                }
                else
                {
                    deltaC = state.Network.Layers[i + 1].Weights.TransposeThisAndMultiply(state.Errors[i + 1]);
                }

                state.Errors[i] = ElementWiseMultiply(deltaC, prime);
            }
        }

        public static LayerGradient CalculateLayerGradients(Vector<double> childActivations, Vector<double> layerError)
        {
            LayerGradient result = new LayerGradient();
            result.WeightGrade = layerError.ToColumnMatrix() * childActivations.ToRowMatrix();
            result.BiaseGrade = layerError.Clone();

            return result;
        }

        public static Vector<double> ElementWiseMultiply(Vector<double> a, Vector<double> b)
        {
            if(a.Count != b.Count)
            {
                throw new Exception("Vectors are not the same size.");
            }

            Vector<double> result = Vector<double>.Build.Dense(a.Count);
            a.MapIndexed((i, v) => v * b[i], result);
            return result;
        }
    }
}
