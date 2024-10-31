using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.MNIST
{
    public class MNISTTestImage
    {
        public byte Label { get; set; }

        public Vector<double> FlattenedImage { get; set; }

        public Vector<double> LabelVector { get; set; }

        public int Width { get; set; }

        public int Height { get; set; }
    }
}
