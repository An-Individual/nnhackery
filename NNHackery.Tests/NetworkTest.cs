using NNHackery.Components;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.Tests
{
    internal class NetworkTest
    {
        [Test]
        public void Sigmoid_Half_CorrectResult()
        {
            double result = Network.Activation.Functions.Sigmoid(0.5);

            Assert.That(result, Is.EqualTo(0.62245933120185459));
        }

        [Test]
        public void DSigmoid_Half_CorrectResult()
        {
            double result = Network.Activation.Derivatives.DSigmoid(0.5);

            Assert.That(result, Is.EqualTo(0.23500371220159449));
        }

        [Test]
        public void ApplyNetwork_LinearNetwork_CorrectResult()
        {
            Network network = new Network(1, 1, 1);

            Assert.That(network.Layers.Count, Is.EqualTo(2));

            network.Layers[0].Weights[0, 0] = 0.5;
            network.Layers[1].Weights[0, 0] = 0.5;

            Vector<double> vector = Vector<double>.Build.Dense(new double[]{ 1 });

            Vector<double> result = network.ApplyNetwork(vector);

            Assert.That(result[0], Is.EqualTo(0.57718538014465226));
        }
    }
}
