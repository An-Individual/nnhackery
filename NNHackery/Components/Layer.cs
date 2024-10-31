using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data.SqlTypes;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using NNHackery.LinearAlgebra;

namespace NNHackery.Components
{
    public class Layer
    {
        public static double GetWeight(Matrix matrix, int sourceNodeIndex, int targetNodeIndex)
        {
            return matrix[sourceNodeIndex, targetNodeIndex];
        }

        public static void SetWeight(Matrix matrix, int sourceNodeIndex, int targetNodeIndex, double value)
        {
            matrix[sourceNodeIndex, targetNodeIndex] = value;
        }

        public static Layer MakeLayer(int oldNodeCount, int newNodeCount)
        {
            Matrix weights = new Matrix(oldNodeCount, newNodeCount);
            Vector biases = new Vector(newNodeCount);

            return new Layer(weights, biases);
        }

        private static double[] MakeRandomArray(int count)
        {
            double[] result = new double[count];
            Parallel.For(0, count, i =>
            {
                result[i] = Random.Shared.NextDouble();
            });
            return result;
        }

        private Layer(Matrix weights, Vector biases)
        {
            if (weights.Height != biases.Size)
            {
                throw new Exception("There are a different number of biases and weight rows.");
            }

            Weights = weights;
            Biases = biases;
        }

        public Matrix Weights { get; }

        public Vector Biases { get; }

        public Vector ApplyToVector(Vector vector, Func<double, double>? activationFunc = null)
        {
            if(vector == null)
            {
                throw new ArgumentNullException(nameof(vector));
            }

            if(vector.Size != Weights.Width)
            {
                throw new Exception("Vector has incorrect number of values");
            }

            Vector result = Matrix.DotProduct(Weights, vector);
            result.Add(Biases);

            if(activationFunc != null)
            {
                result.ApplyElementwiseFunction(activationFunc);
            }

            return result;
        }

        #region Serialization

        public static Layer Deserialize(byte[] data)
        {
            using (MemoryStream stream = new MemoryStream(data))
            {
                return Deserialize(stream);
            }
        }

        public static Layer Deserialize(Stream stream)
        {
            Matrix weights = ReadMatrix(stream);
            Vector biases = new Vector(ReadMatrix(stream));

            return new Layer(weights, biases);
        }

        private static Matrix ReadMatrix(Stream stream)
        {
            byte[] intBuffer = new byte[sizeof(int)];

            stream.ReadExactly(intBuffer);
            int width = BitConverter.ToInt32(intBuffer);

            stream.ReadExactly(intBuffer);
            int height = BitConverter.ToInt32(intBuffer);

            Matrix result = new Matrix(width, height);
            byte[] doubleBuffer = new byte[sizeof(double)];

            for (int y = 0; y < result.Height; y++)
            {
                for (int x = 0; x < result.Width; x++)
                {
                    stream.ReadExactly(doubleBuffer);
                    result[x,y] = BitConverter.ToDouble(doubleBuffer);
                }
            }

            return result;
        }

        public byte[] Serialize()
        {
            using (MemoryStream stream = new MemoryStream())
            {
                Serialize(stream);
                return stream.ToArray();
            }
        }

        public void Serialize(Stream stream)
        {
            SerializeMatrix(stream, Weights);
            SerializeMatrix(stream, Biases.Matrix);
        }

        private static void SerializeMatrix(Stream stream, Matrix matrix)
        {
            stream.Write(BitConverter.GetBytes(matrix.Width));
            stream.Write(BitConverter.GetBytes(matrix.Height));

            for (int y = 0; y < matrix.Height; y++)
            {
                for (int x = 0; x < matrix.Width; x++)
                {
                    stream.Write(BitConverter.GetBytes(matrix[x,y]));
                }
            }
        }

        #endregion Serialization
    }
}
