using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.LinearAlgebra
{
    /// <summary>
    /// This implementation uses 1 dimenional arrays because
    /// it saves a non-trivial amount of time when loading
    /// test data.
    /// </summary>
    public class Matrix
    {
        public static Matrix DotProduct(Matrix a, Matrix b)
        {
            if (a == null || b == null)
            {
                throw new ArgumentNullException();
            }

            if (a.Width != b.Height)
            {
                throw new Exception("Matrix B is the wrong width to do a dot product with matrix A.");
            }

            Matrix result = new Matrix(b.Width, a.Height);

            // For every column in B
            Parallel.For(0, b.Width, bCol =>
            {
                // For every row in A
                Parallel.For(0, a.Height, aRow =>
                {
                    // Multiply all the values in the row
                    // and add them together to get the value
                    // for this cell of the final matrix.
                    double[] values = new double[a.Width];
                    Parallel.For(0, values.Length, i =>
                    {
                        values[i] = a[i, aRow] * b[bCol, i];
                    });

                    result[bCol, aRow] = values.Aggregate((c,v) => c + v);
                });
            });

            return result;
        }

        public static Vector DotProduct(Matrix m, Vector v)
        {
            return new Vector(DotProduct(m, v.Matrix));
        }

        public Matrix(int width, int height)
        {
            if(width <= 0 || height <= 0)
            {
                throw new Exception("Cannot create empty matrices.");
            }

            _width = width;
            _height = height;
            _values = new double[width * height];

            Transposed = false;
        }

        public Matrix(int width, double[] values)
        {
            if (width <= 0 || values.Length == 0)
            {
                throw new Exception("Cannot create empty matrices.");
            }

            if(values.Length % width != 0)
            {
                throw new Exception("Matrix would not be square.");
            }

            _width = width;
            _height = values.Length / _width;
            _values = values;
        }

        private readonly int _width;
        private readonly int _height;
        private readonly double[] _values;

        public bool Transposed { get; private set; }

        public int Width
        {
            get { return Transposed ? _height : _width; }
        }

        public int Height
        {
            get { return Transposed ? _width : _height; }
        }

        private int GetIndex(int x, int y)
        {
            if (x < 0 || y < 0)
            {
                throw new IndexOutOfRangeException();
            }

            if (Transposed)
            {
                if (x >= _height || y >= _width)
                {
                    throw new IndexOutOfRangeException();
                }

                return x * _width + y;
            }
            else
            {
                if (x >= _width || y >= _height)
                {
                    throw new IndexOutOfRangeException();
                }

                return y * _width + x;
            }
        }

        public double this[int x, int y]
        {
            get
            {
                return _values[GetIndex(x, y)];
            }
            set
            {
                _values[GetIndex(x, y)] = value;
            }
        }

        public void ApplyElementwiseFunction(Func<double, double> function)
        {
            Parallel.For(0, Width, x =>
            {
                Parallel.For(0, Height, y =>
                {
                    this[x, y] = function(this[x, y]);
                });
            });
        }

        public void ApplyElementwiseMatrixFunction(Matrix matrix, Func<double, double, double> function)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (matrix.Width != Width || matrix.Height != Height)
            {
                throw new Exception("Cannot perform an elementwise function on matrices of different sizes.");
            }

            Parallel.For(0, Width, x =>
            {
                Parallel.For(0, Height, y =>
                {
                    this[x, y] = function(this[x,y], matrix[x, y]);
                });
            });
        }

        public void Add(Matrix matrix)
        {
            ApplyElementwiseMatrixFunction(matrix, (c, n) => c + n);
        }

        /// <summary>
        /// Returns a new wrapper of the matrix
        /// values, but transposed. THIS DOES
        /// NOT CREATE A CLONE.
        /// </summary>
        /// <returns></returns>
        public Matrix Transpose()
        {
            Matrix result = new Matrix(_width, _values);
            result.Transposed = true;
            return result;
        }

        public Matrix Clone()
        {
            double[] values = new double[_values.Length];
            _values.CopyTo(values, 0);

            Matrix result = new Matrix(_width, values);
            result.Transposed = Transposed;

            return result;
        }
    }
}
