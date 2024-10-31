using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.LinearAlgebra
{
    public class Vector
    {
        public static Vector ElementwiseCombine(Vector a, Vector b, Func<double, double, double> operation)
        {
            if (a == null)
            {
                throw new ArgumentNullException(nameof(a));
            }

            if (b == null)
            {
                throw new ArgumentNullException(nameof(b));
            }

            if (a.Size != b.Size)
            {
                throw new Exception("Vectors have a different number of elements.");
            }

            double[] values = new double[a.Size];
            Parallel.For(0, values.Length, i =>
            {
                values[i] = operation(a[i], b[i]);
            });

            return new Vector(values);
        }

        public Vector(int size) : this(new Matrix(1, size))
        {
        }

        public Vector(double[] values) : this(new Matrix(1, values))
        {
        }

        public Vector(Matrix matrix)
        {
            if(matrix == null)
            {
                throw new ArgumentNullException(nameof (matrix));
            }

            if (matrix.Width != 1)
            {
                throw new Exception("Matrix has more than 1 column.");
            }

            if (matrix.Height == 0)
            {
                throw new Exception("Matrix is empty.");
            }

            Matrix = matrix;
        }

        public Matrix Matrix { get; }

        public double this[int index]
        {
            get
            {
                return Matrix[0, index];
            }
            set
            {
                Matrix[0, index] = value;
            }
        }

        public int Size
        {
            get
            {
                return Matrix.Height;
            }
        }

        public void ApplyElementwiseFunction(Func<double, double> function)
        {
            Parallel.For(0, Size, i =>
            {
                this[i] = function(this[i]);
            });
        }

        public void Add(Vector vector)
        {
            Matrix.Add(vector.Matrix);
        }

        public Vector Clone()
        {
            return new Vector(Matrix.Clone());
        }

        /// <summary>
        /// Returns a new state reflecting the current state with
        /// SoftMax run over it's values.
        /// </summary>
        /// <returns></returns>
        public Vector Softmax(double temperature = 1)
        {
            if (temperature <= 0)
            {
                throw new Exception("Temperature must be a postive, non-zero double.");
            }

            double[] result = new double[Size];
            Parallel.For(0, Size, i =>
            {
                result[i] = Math.Exp(this[i] / temperature);
            });

            double total = result.Aggregate((c, v) => c + v);

            Parallel.For(0, Size, i =>
            {
                result[i] = result[i] / total;
            });

            return new Vector(result);
        }
    }
}
