using MathNet.Numerics.LinearAlgebra;
using NNHackery.Components;
using NNHackery.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNHackery.Tests
{
    internal class QuadraticTrainerTests
    {
        [Test]
        public void CalculateVectorsAndActivations_SingleLineNetwork_CorrectValues()
        {
            Network network = new Network(1, 1, 1);

            network.Layers[0].Weights[0, 0] = 0.5;
            network.Layers[1].Weights[0, 0] = 0.2;

            Vector<double> vector = Vector<double>.Build.Dense([1]);

            QuadraticTrainer.IterationState state = new QuadraticTrainer.IterationState();
            state.Network = network;
            state.Input = vector;

            QuadraticTrainer.CalculateVectorsAndActivations(state);

            Assert.That(state.Vectors.Count, Is.EqualTo(2));
            Assert.That(state.Activations.Count, Is.EqualTo(2));

            Assert.That(state.Vectors[0][0], Is.EqualTo(0.5));
            Assert.That(state.Vectors[1][0], Is.EqualTo(0.12449186624037092));

            Assert.That(state.Activations[0][0], Is.EqualTo(0.62245933120185459));
            Assert.That(state.Activations[1][0], Is.EqualTo(0.53108283286480273));
        }

        [Test]
        public void CalculateErrors_SingleLineNetwork_CorrectValues()
        {
            Network network = new Network(1, 1, 1);

            network.Layers[0].Weights[0, 0] = 0.5;
            network.Layers[1].Weights[0, 0] = 0.2;

            Vector<double> vector = Vector<double>.Build.Dense([1]);

            QuadraticTrainer.IterationState state = new QuadraticTrainer.IterationState();
            state.Network = network;
            state.Input = vector;
            state.Vectors = [
                Vector<double>.Build.Dense([0.5]),
                Vector<double>.Build.Dense([0.12449186624037092]),
            ];
            state.Activations = [
                Vector<double>.Build.Dense([0.62245933120185459]),
                Vector<double>.Build.Dense([0.53108283286480273]),
            ];
            state.Expected = Vector<double>.Build.Dense([1]);

            QuadraticTrainer.CalculateErrors(state);

            Assert.That(state.Errors.Length, Is.EqualTo(2));

            Assert.That(state.Errors[1][0], Is.EqualTo(-0.11677625098016561));
            Assert.That(state.Errors[0][0], Is.EqualTo(-0.0054885704954648019));
        }

        [Test]
        public void CalculateGradients_SingleLineNetwork_CorrectValues()
        {
            Network network = new Network(1, 1, 1);

            network.Layers[0].Weights[0, 0] = 0.5;
            network.Layers[1].Weights[0, 0] = 0.2;

            Vector<double> input = Vector<double>.Build.Dense([1]);
            Vector<double> expected = Vector<double>.Build.Dense([1]);

            QuadraticTrainer.LayerGradient[] gradients = QuadraticTrainer.CalculateGradients(network, input, expected);

            Assert.That(gradients.Length, Is.EqualTo(2));

            Assert.That(gradients[1].BiaseGrade[0], Is.EqualTo(-0.11677625098016561));
            Assert.That(gradients[1].WeightGrade[0, 0], Is.EqualTo(-0.0726884670853738));

            Assert.That(gradients[0].BiaseGrade[0], Is.EqualTo(-0.0054885704954648019));
            Assert.That(gradients[0].WeightGrade[0, 0], Is.EqualTo(-0.0054885704954648019));
        }

        [Test]
        public void RunGradientDescent_SingleLineNetwork_CorrectValues()
        {
            Network network = new Network(1, 1, 1);

            network.Layers[0].Weights[0, 0] = 0.5;
            network.Layers[1].Weights[0, 0] = 0.2;

            Vector<double> input = Vector<double>.Build.Dense([1]);
            Vector<double> expected = Vector<double>.Build.Dense([1]);

            QuadraticTrainer trainer = new QuadraticTrainer();
            trainer.RunGradientDescent(network, [input], [expected], 1);

            Assert.That(network.Layers[0].Weights[0, 0], Is.EqualTo(0.5054885704954648019));
            Assert.That(network.Layers[0].Biases[0], Is.EqualTo(0.0054885704954648019));
            Assert.That(network.Layers[1].Weights[0, 0], Is.EqualTo(0.2726884670853738));
            Assert.That(network.Layers[1].Biases[0], Is.EqualTo(0.11677625098016561));
        }

        [TestCase]
        public void TraningRun_AddCaseWhenEmpty_CaseAdded()
        {
            Network network = new Network(1, 1);

            QuadraticTrainer.TrainingRun run = new QuadraticTrainer.TrainingRun(network);
            run.AddCase([new QuadraticTrainer.LayerGradient() {
                WeightGrade = Matrix<double>.Build.Dense(1, 1, 0.5),
                BiaseGrade = Vector<double>.Build.Dense([0.2])
            }]);

            Assert.That(run.Gradients[0].WeightGrade[0, 0], Is.EqualTo(0.5));
            Assert.That(run.Gradients[0].BiaseGrade[0], Is.EqualTo(0.2));
        }



        [TestCase]
        public void TraningRun_AddCaseWhenFilled_ValuesIncremented()
        {
            Network network = new Network(1, 1);

            QuadraticTrainer.TrainingRun run = new QuadraticTrainer.TrainingRun(network);

            run.Gradients[0] = new QuadraticTrainer.LayerGradient()
            {
                WeightGrade = Matrix<double>.Build.Dense(1, 1, 2),
                BiaseGrade = Vector<double>.Build.Dense([3])
            };

            run.AddCase([new QuadraticTrainer.LayerGradient() {
                WeightGrade = Matrix<double>.Build.Dense(1, 1, 0.5),
                BiaseGrade = Vector<double>.Build.Dense([0.2])
            }]);

            Assert.That(run.Gradients[0].WeightGrade[0, 0], Is.EqualTo(2.5));
            Assert.That(run.Gradients[0].BiaseGrade[0], Is.EqualTo(3.2));
        }
    }
}
