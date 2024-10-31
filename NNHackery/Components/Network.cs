using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNHackery.LinearAlgebra;

namespace NNHackery.Components
{
    public class Network
    {
        public static class Activation
        {
            public static class Functions
            {
                public static double Sigmoid(double value)
                {
                    return 1 / (1 + Math.Exp(-1 * value));
                }

                // Rectificed Linear Unit
                public static double ReLU(double value)
                {
                    return Math.Max(0, value);
                }

                // Gaussian Error Linear Unit
                public static double GELU(double value)
                {
                    // Pulled from https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
                    return 0.5 * value * (
                        1 + Math.Tanh(
                            Math.Sqrt(2 / Math.PI) * (
                                value + 0.044715 * Math.Pow(value, 3)
                            )
                            )
                        );
                }
            }

            public static class Derivatives
            {
                // Pulled from: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
                public static double DSigmoid(double value)
                {
                    return Functions.Sigmoid(value) * (1 - Functions.Sigmoid(value));
                }
            }
        }

        private static Layer[] MakeLayers(int[] nodeCounts)
        {
            if(nodeCounts.Length < 1)
            {
                throw new Exception("There must be at least 2 node layers.");
            }

            if (nodeCounts.Any(c => c <= 0))
            {
                throw new Exception("All layers must have at least 1 node.");
            }

            Layer[] result = new Layer[nodeCounts.Length - 1];
            for(int i = 0; i < result.Length; i++)
            {
                result[i] = Layer.MakeLayer(nodeCounts[i], nodeCounts[i+1]);
            }

            return result;
        }

        public Network(params int[] nodeCounts) : this(MakeLayers(nodeCounts))
        {
        }

        public Network(Layer[] layers) 
        {
            if(layers == null)
            {
                throw new ArgumentNullException(nameof(layers));
            }

            Layers = layers;

            ActivationFunction = Activation.Functions.Sigmoid;
            ActivationDerivative = Activation.Derivatives.DSigmoid;
        }

        public Func<double, double> ActivationFunction { get; set; }

        public Func<double, double> ActivationDerivative { get; set; }

        public Layer[] Layers { get; }

        public Vector ApplyNetwork(Vector inputState)
        {
            Vector currentState = inputState;
            for(int i = 0; i < Layers.Length; i++)
            {
                currentState = Layers[i].ApplyToVector(currentState, ActivationFunction);
            }

            return currentState;
        }

        #region Serialization

        public static Network Deserialize(byte[] data)
        {
            using(MemoryStream stream = new MemoryStream(data))
            {
                return ReadNetwork(stream);
            }
        }

        public static Network ReadNetwork(Stream stream)
        {
            byte[] intBuffer = new byte[sizeof(int)];
            stream.ReadExactly(intBuffer);
            int stepCount = BitConverter.ToInt32(intBuffer);

            Layer[] steps = new Layer[stepCount];
            for(int i = 0; i < stepCount; i++)
            {
                steps[i] = Layer.Deserialize(stream);
            }

            return new Network(steps);
        }

        public byte[] Serialize()
        {
            using(MemoryStream stream = new MemoryStream())
            {
                Serialize(stream);
                return stream.ToArray();
            }
        }

        public void Serialize(Stream stream)
        {
            stream.Write(BitConverter.GetBytes(Layers.Count()));
            foreach (Layer step in Layers)
            {
                step.Serialize(stream);
            }
        }

        #endregion Serialization
    }
}
