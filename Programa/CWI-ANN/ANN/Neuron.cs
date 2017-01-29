using System;
using System.Collections.Generic;
using System.Linq;

namespace CWIANN
{
	public class Neuron
	{
		public double Output { get; set; }
		public List<double> Weight { get; private set; }
		public double Gradient { get; private set; }
		private double _euler = 2.718281828;
		private double _learningRate = 0.35;

		public Neuron(int inputs, Random rnd)
		{
			Weight = new List<double>();
			for (int i = 0; i < inputs; i++)
				Weight.Add(rnd.NextDouble() * 2 - 1);
		}

		public double Activate(double result)
		{
			Output = (1 / (1 + Math.Pow(_euler, -result)));
			return Output;
		}

		public double LoadNeuron(List<double> input)
		{
			if (input.Count == Weight.Count)
				return Weight.Select((weight, i) => input[i] * weight).Sum();
			else
				throw new Exception("Existe diferença entre os Inputs do neurônio e os pesos");
		}

		public void UpdateWeightsOutputLayer(double output, List<double> outputLeft, double target)
		{
			Gradient = output * ((1 - output) * (target - output));
			for (int i = 0; i < Weight.Count; i++)
				Weight[i] = Weight[i] + _learningRate * Gradient * outputLeft[i];
		}

		public void UpdateWeightsHiddenLayer(double output, List<double> outputLeft, List<double> weightRight, List<double> gradientRight)
		{
			Gradient = output * ((1 - output) * (weightRight.Select((weight, i) => gradientRight[i] * weight).Sum()));
			for (int i = 0; i < Weight.Count; i++)
				Weight[i] = Weight[i] + _learningRate * Gradient * outputLeft[i];

		}
	}
}

