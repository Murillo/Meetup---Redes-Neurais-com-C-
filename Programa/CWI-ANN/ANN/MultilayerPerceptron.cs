using System;
using System.Collections.Generic;
using System.Linq;

namespace CWIANN
{
	public class MultilayerPerceptron
	{
		public List<Layer> Layers { get; set; }
		public List<double> Inputs { get; set; }
		public List<double> Outputs { get; set; }
		private double _error = 0.0001d;

		public MultilayerPerceptron(int input, int hidden, int output)
		{
			Layers = new List<Layer>();
			Inputs = new List<double>();
			Outputs = new List<double>();

			// hidden layer
			Layers.Add(new Layer(input, hidden));
			// output layer
			Layers.Add(new Layer(hidden, output));
		}

		public double[] Run(double[] inputs)
		{
			foreach (Layer layer in Layers)
				inputs = CalculateLayer(layer, inputs);
			return inputs;
		}

		public void Training(double[,] inputs, double[] outputs)
		{
			long iteration = 0;
			double error;
			do
			{
				error = 0;
				int numberTrain = inputs.Length / inputs.GetLength(1);
				for (int i = 0; i < numberTrain; i++)
				{
					double tagert = outputs[i];
					double[] inputTest = new double[inputs.GetLength(1)];
					for (int j = 0; j < inputs.GetLength(1); j++)
						inputTest[j] = inputs[i, j];

					double output = Run(inputTest).FirstOrDefault(k => k > 0);
					Backpropagation(inputTest, output, tagert);

					double delta = tagert - output;
					error += Math.Pow(delta, 2);
					System.Diagnostics.Debug.WriteLine(Math.Round((output * 100) / tagert, 2) + "%");
				}

				iteration++;
			} while (error >= _error);

			System.Diagnostics.Debug.WriteLine("Quantidade de iterações: " + iteration);
		}

		private double[] CalculateLayer(Layer layer, double[] input)
		{
			double[] values = new double[layer.Neurons.Count];
			for (int i = 0; i < layer.Neurons.Count; i++)
			{
				var valueNeuron = layer.Neurons[i].LoadNeuron(new List<double>(input));
				values[i] = layer.Neurons[i].Activate(valueNeuron);
			}
			return values;
		}

		private void Backpropagation(double[] input, double output, double target)
		{
			// Go layers
			for (int i = (Layers.Count - 1); i >= 0; i--)
			{
				// Go neurons layer
				for (int j = 0; j < Layers[i].Neurons.Count; j++)
				{
					List<double> outputsLeft = new List<double>();
					outputsLeft.AddRange(i > 0 ? Layers[i - 1].Neurons.Select(t => t.Output) : input);

					if (i == Layers.Count - 1)
					{
						Layers[i].Neurons[j].UpdateWeightsOutputLayer(output, outputsLeft, target);
					}
					else
					{
						Layers[i].Neurons[j].UpdateWeightsHiddenLayer(
							Layers[i].Neurons[j].Output,
							outputsLeft,
							Layers[i + 1].Neurons.Select(k => k.Weight[j]).ToList(),
							Layers[i + 1].Neurons.Select(k => k.Gradient).ToList());
					}
				}
			}
		}
	}
}

