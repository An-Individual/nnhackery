using MathNet.Numerics.LinearAlgebra;
using NNHackery.Components;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.Trainers
{
    internal interface ITrainer
    {
        void RunGradientDescent(Network network, Vector<double>[] inputs, Vector<double>[] expectedOutputs, double learningRate);
    }
}
