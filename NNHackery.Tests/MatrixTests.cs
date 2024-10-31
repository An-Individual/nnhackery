using NNHackery.LinearAlgebra;

namespace NNHackery.Tests
{
    public class MatrixTests
    {
        [Test]
        public void Constructor_ZeroWidth_Throws()
        {
            Assert.Throws(typeof(Exception), () => new Matrix(0, 1));
        }

        [Test]
        public void Constructor_ZeroHeight_Throws()
        {
            Assert.Throws(typeof(Exception), () => new Matrix(1, 0));
        }

        [Test]
        public void Constructor_ZeroWidthWithArray_Throws()
        {
            Assert.Throws(typeof(Exception), () => new Matrix(0, [1]));
        }

        [Test]
        public void Constructor_EmptyArray_Throws()
        {
            Assert.Throws(typeof(Exception), () => new Matrix(1, []));
        }


        [Test]
        public void Constructor_EvenWidthWithOddArray_Throws()
        {
            Assert.Throws(typeof(Exception), () => new Matrix(2, [1, 2, 3]));
        }

        [Test]
        public void Constructor_1By1_WidthHeightAndDefaultsCorrect()
        {
            Matrix matrix = new Matrix(1, 1);
            
            Assert.That(matrix.Width, Is.EqualTo(1));
            Assert.That(matrix.Height, Is.EqualTo(1));
            Assert.That(matrix[0, 0], Is.EqualTo(0));
        }

        [Test]
        public void Constructor_1By2_WidthHeightAndDefaultsCorrect()
        {
            Matrix matrix = new Matrix(1, 2);

            Assert.That(matrix.Width, Is.EqualTo(1));
            Assert.That(matrix.Height, Is.EqualTo(2));
            Assert.That(matrix[0, 0], Is.EqualTo(0));
            Assert.That(matrix[0, 1], Is.EqualTo(0));
        }

        [Test]
        public void Constructor_2By1_WidthHeightAndDefaultsCorrect()
        {
            Matrix matrix = new Matrix(2, 1);

            Assert.That(matrix.Width, Is.EqualTo(2));
            Assert.That(matrix.Height, Is.EqualTo(1));
            Assert.That(matrix[0, 0], Is.EqualTo(0));
            Assert.That(matrix[1, 0], Is.EqualTo(0));
        }

        [Test]
        public void Constructor_1By1WithArray_WidthHeightAndValuesCorrect()
        {
            Matrix matrix = new Matrix(1, [3]);

            Assert.That(matrix.Width, Is.EqualTo(1));
            Assert.That(matrix.Height, Is.EqualTo(1));
            Assert.That(matrix[0, 0], Is.EqualTo(3));
        }

        [Test]
        public void Constructor_1By2WithArray_WidthHeightAndValuesCorrect()
        {
            Matrix matrix = new Matrix(1, [3, 4]);

            Assert.That(matrix.Width, Is.EqualTo(1));
            Assert.That(matrix.Height, Is.EqualTo(2));
            Assert.That(matrix[0, 0], Is.EqualTo(3));
            Assert.That(matrix[0, 1], Is.EqualTo(4));
        }

        [Test]
        public void Constructor_2By1WithArray_WidthHeightAndValuesCorrect()
        {
            Matrix matrix = new Matrix(2, [3, 4]);

            Assert.That(matrix.Width, Is.EqualTo(2));
            Assert.That(matrix.Height, Is.EqualTo(1));
            Assert.That(matrix[0, 0], Is.EqualTo(3));
            Assert.That(matrix[1, 0], Is.EqualTo(4));
        }

        [Test]
        public void Getter_OutOfBounds_Throws()
        {
            Matrix matrix = new Matrix(2, 3);

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[2, 0];
            });

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[-1, 0];
            });

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[0, 3];
            });

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[0, -1];
            });
        }

        [Test]
        public void Getter_OutOfBoundsTransposed_Throws()
        {
            Matrix matrix = new Matrix(2, 3);
            matrix = matrix.Transpose();

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[3, 0];
            });

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[-1, 0];
            });

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[0, 2];
            });

            Assert.Throws(typeof(IndexOutOfRangeException), () =>
            {
                double value = matrix[0, -1];
            });
        }

        [Test]
        public void Getter_3By2_AllValuesCorrect()
        {
            Matrix matrix = new Matrix(3, [1, 2, 3, 4, 5, 6]);

            Assert.That(matrix.Width, Is.EqualTo(3));
            Assert.That(matrix.Height, Is.EqualTo(2));
            Assert.That(matrix[0, 0], Is.EqualTo(1));
            Assert.That(matrix[1, 0], Is.EqualTo(2));
            Assert.That(matrix[2, 0], Is.EqualTo(3));
            Assert.That(matrix[0, 1], Is.EqualTo(4));
            Assert.That(matrix[1, 1], Is.EqualTo(5));
            Assert.That(matrix[2, 1], Is.EqualTo(6));
        }

        [Test]
        public void Getter_3By2Transposed_AllValuesCorrect()
        {
            Matrix matrix = new Matrix(3, [1, 2, 3, 4, 5, 6]);
            matrix = matrix.Transpose();

            Assert.That(matrix.Width, Is.EqualTo(2));
            Assert.That(matrix.Height, Is.EqualTo(3));
            Assert.That(matrix[0, 0], Is.EqualTo(1));
            Assert.That(matrix[0, 1], Is.EqualTo(2));
            Assert.That(matrix[0, 2], Is.EqualTo(3));
            Assert.That(matrix[1, 0], Is.EqualTo(4));
            Assert.That(matrix[1, 1], Is.EqualTo(5));
            Assert.That(matrix[1, 2], Is.EqualTo(6));
        }

        [Test]
        public void Setter_3By2_AllValuesCorrect()
        {
            Matrix matrix = new Matrix(3, 2);

            matrix[0, 0] = 1;
            matrix[1, 0] = 2;
            matrix[2, 0] = 3;
            matrix[0, 1] = 4;
            matrix[1, 1] = 5;
            matrix[2, 1] = 6;

            Assert.That(matrix[0, 0], Is.EqualTo(1));
            Assert.That(matrix[1, 0], Is.EqualTo(2));
            Assert.That(matrix[2, 0], Is.EqualTo(3));
            Assert.That(matrix[0, 1], Is.EqualTo(4));
            Assert.That(matrix[1, 1], Is.EqualTo(5));
            Assert.That(matrix[2, 1], Is.EqualTo(6));
        }

        [Test]
        public void Setter_3By2Transposed_AllValuesCorrect()
        {
            Matrix matrix = new Matrix(3, 2);
            matrix.Transpose();

            matrix[0, 0] = 1;
            matrix[1, 0] = 2;
            matrix[2, 0] = 3;
            matrix[0, 1] = 4;
            matrix[1, 1] = 5;
            matrix[2, 1] = 6;

            Assert.That(matrix[0, 0], Is.EqualTo(1));
            Assert.That(matrix[1, 0], Is.EqualTo(2));
            Assert.That(matrix[2, 0], Is.EqualTo(3));
            Assert.That(matrix[0, 1], Is.EqualTo(4));
            Assert.That(matrix[1, 1], Is.EqualTo(5));
            Assert.That(matrix[2, 1], Is.EqualTo(6));
        }

        [Test]
        public void DotProduct_11by11_1by1ValueCorrect()
        {
            Matrix a = new Matrix(1, [2]);
            Matrix b = new Matrix(1, [3]);

            Matrix result = Matrix.DotProduct(a, b);

            Assert.That(result.Width, Is.EqualTo(1));
            Assert.That(result.Height, Is.EqualTo(1));
            Assert.That(result[0, 0], Is.EqualTo(6));
        }

        [Test]
        public void DotProduct_22by12_1by2ValuesCorrect()
        {
            Matrix a = new Matrix(2, [1, 2, 3, 4]);
            Matrix b = new Matrix(1, [1, 2]);

            Matrix result = Matrix.DotProduct(a, b);

            Assert.That(result.Width, Is.EqualTo(1));
            Assert.That(result.Height, Is.EqualTo(2));
            Assert.That(result[0, 0], Is.EqualTo(5));
            Assert.That(result[0, 1], Is.EqualTo(11));
        }

        [Test]
        public void DotProduct_12by21_2by2ValuesCorrect()
        {
            Matrix a = new Matrix(1, [1, 2]);
            Matrix b = new Matrix(2, [3, 4]);

            Matrix result = Matrix.DotProduct(a, b);

            Assert.That(result.Width, Is.EqualTo(2));
            Assert.That(result.Height, Is.EqualTo(2));
            Assert.That(result[0, 0], Is.EqualTo(3));
            Assert.That(result[1, 0], Is.EqualTo(4));
            Assert.That(result[0, 1], Is.EqualTo(6));
            Assert.That(result[1, 1], Is.EqualTo(8));
        }
    }
}