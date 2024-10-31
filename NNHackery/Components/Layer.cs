using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data.SqlTypes;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.Components
{
    public class Layer
    {
        public static Layer MakeLayer(int oldNodeCount, int newNodeCount)
        {
            Matrix<double> weights = Matrix.Build.Dense(newNodeCount, oldNodeCount);
            Vector<double> biases = Vector.Build.Dense(newNodeCount);

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

        private Layer(Matrix<double> weights, Vector<double> biases)
        {
            if (weights.RowCount != biases.Count)
            {
                throw new Exception("There are a different number of biases and weight rows.");
            }

            Weights = weights;
            Biases = biases;
        }

        public Matrix<double> Weights { get; }

        public Vector<double> Biases { get; }

        public Vector<double> ApplyToVector(Vector<double> vector, Func<double, double>? activationFunc = null)
        {
            if(vector == null)
            {
                throw new ArgumentNullException(nameof(vector));
            }

            if(vector.Count != Weights.ColumnCount)
            {
                throw new Exception("Vector has incorrect number of values");
            }

            Vector<double> result = Weights * vector + Biases;

            if(activationFunc != null)
            {
                result.MapInplace(activationFunc);
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
            Matrix<double> weights = ReadMatrix(stream);
            Vector<double> biases = ReadVector(stream);

            return new Layer(weights, biases);
        }

        private static Matrix<double> ReadMatrix(Stream stream)
        {
            byte[] intBuffer = new byte[sizeof(int)];

            stream.ReadExactly(intBuffer);
            int columns = BitConverter.ToInt32(intBuffer);

            stream.ReadExactly(intBuffer);
            int rows = BitConverter.ToInt32(intBuffer);

            Matrix<double> result = Matrix<double>.Build.Dense(rows, columns);
            byte[] doubleBuffer = new byte[sizeof(double)];

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    stream.ReadExactly(doubleBuffer);
                    result[r, c] = BitConverter.ToDouble(doubleBuffer);
                }
            }

            return result;
        }

        private static Vector<double> ReadVector(Stream stream)
        {
            byte[] intBuffer = new byte[sizeof(int)];

            stream.ReadExactly(intBuffer);
            int count = BitConverter.ToInt32(intBuffer);

            Vector<double> result = Vector<double>.Build.Dense(count);
            byte[] doubleBuffer = new byte[sizeof(double)];

            for (int i = 0; i < count; i++)
            {
                stream.ReadExactly(doubleBuffer);
                result[i] = BitConverter.ToDouble(doubleBuffer);
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
            SerializeVector(stream, Biases);
        }

        private static void SerializeMatrix(Stream stream, Matrix<double> matrix)
        {
            stream.Write(BitConverter.GetBytes(matrix.ColumnCount));
            stream.Write(BitConverter.GetBytes(matrix.RowCount));

            for (int row = 0; row < matrix.RowCount; row++)
            {
                for (int col = 0; col < matrix.ColumnCount; col++)
                {
                    stream.Write(BitConverter.GetBytes(matrix[row, col]));
                }
            }
        }

        private static void SerializeVector(Stream stream, Vector<double> vector)
        {
            stream.Write(BitConverter.GetBytes(vector.Count));

            for (int i = 0; i < vector.Count; i++)
            {
                stream.Write(BitConverter.GetBytes(vector[i]));
            }
        }

        #endregion Serialization
    }
}
