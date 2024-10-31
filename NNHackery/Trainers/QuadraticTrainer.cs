using NNHackery.Components;
using NNHackery.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.XPath;

namespace NNHackery.Trainers
{
    public static class QuadraticTrainer
    {
        private class IterationState
        {
            public Network Network { get; set; }

            public Vector Input { get; set; }

            public Vector Expected { get; set; }

            public Vector[] Vectors { get; set; }

            public Vector[] Activations { get; set; }

            public Vector[] Errors { get; set; }
        }

        private class LayerGradient
        {
            public Matrix WeightGrade { get; set; }

            public Vector BiaseGrade { get; set; }

            public void ApplyToLayer(Layer layer, double learningRateOverBatchSize)
            {
                layer.Weights.ApplyElementwiseMatrixFunction(WeightGrade, (w, g) => w - (learningRateOverBatchSize * g));
                layer.Biases.Matrix.ApplyElementwiseMatrixFunction(BiaseGrade.Matrix, (b, g) => b - (learningRateOverBatchSize * g));
            }
        }

        private class TrainingRun
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
                            Gradients[i].WeightGrade.Add(testCase[i].WeightGrade);
                            Gradients[i].BiaseGrade.Add(testCase[i].BiaseGrade);
                        }
                    });
                }
            }
        }

        public static void RunGradientDescent(Network network, Vector[] inputs, Vector[] expectedOutputs, double learningRate)
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

            double learningRateOverBatchSize = learningRate / inputs.Length;

            Parallel.For(0, network.Layers.Length, i =>
            {
                run.Gradients[i].ApplyToLayer(network.Layers[i], learningRateOverBatchSize);
            });
        }

        private static LayerGradient[] CalculateGradients(Network network, Vector input, Vector expected)
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
                Vector activation;
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

        private static void CalculateVectorsAndActivations(IterationState state)
        {
            state.Vectors = new Vector[state.Network.Layers.Length];
            state.Activations = new Vector[state.Network.Layers.Length];

            Vector currentVector = state.Input;

            for (int i = 0; i < state.Network.Layers.Length; i++)
            {
                currentVector = state.Network.Layers[i].ApplyToVector(currentVector);

                state.Vectors[i] = currentVector;
                state.Activations[i] = currentVector.Clone();
                state.Activations[i].ApplyElementwiseFunction(state.Network.ActivationFunction);
                currentVector = state.Activations[i];
            }
        }

        private static void CalculateErrors(IterationState state)
        {
            state.Errors = new Vector[state.Network.Layers.Length];

            int lastIndex = state.Network.Layers.Length - 1;

            state.Errors[lastIndex] = ComputeLastLayerError(state.Vectors[lastIndex], state.Activations[lastIndex], state.Expected, state.Network.ActivationDerivative);

            for (int i = lastIndex - 1; i >= 0; i--)
            {
                state.Errors[i] = ComputeLayerError(state.Vectors[i], state.Network.Layers[i + 1], state.Errors[i + 1], state.Network.ActivationDerivative);
            }
        }

        private static Vector ComputeLastLayerError(Vector vector, Vector activation, Vector expectedOutput, Func<double, double> activationDerivative)
        {
            // Subtract the expected output from the activation (aka, the final output).
            Vector difference = Vector.ElementwiseCombine(activation, expectedOutput, (a, e) => a - e);

            // The result is an elementwise multiplication of the difference vector and the layer's vector with
            // the derivative of the activation function applied.
            return Vector.ElementwiseCombine(difference, vector, (dif, der) => dif * activationDerivative(der));
        }

        private static Vector ComputeLayerError(Vector vector, Layer parentLayer, Vector parentError, Func<double, double> activationDerivative)
        {
            // Dot product of the transpose of the parent layer's weights on the parent's error.
            Vector transposeError = Matrix.DotProduct(parentLayer.Weights.Transpose(), parentError);

            // The result is an elementwise multiplication of the transpose error vector and the layer's vector with
            // the derivative of the activation function applied.
            return Vector.ElementwiseCombine(transposeError, vector, (dif, der) => dif * activationDerivative(der));
        }

        private static LayerGradient CalculateLayerGradients(Vector childActivations, Vector layerError)
        {
            LayerGradient result = new LayerGradient();
            result.WeightGrade = new Matrix(childActivations.Size, layerError.Size);

            Parallel.For(0, childActivations.Size, a =>
            {
                Parallel.For(0, layerError.Size, e =>
                {
                    Layer.SetWeight(result.WeightGrade, a, e, childActivations[a] * layerError[e]);
                });
            });

            result.BiaseGrade = layerError.Clone();

            return result;
        }
    }
}
