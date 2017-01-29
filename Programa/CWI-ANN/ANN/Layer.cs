using System;
using System.Collections.Generic;

namespace CWIANN
{
	public class Layer
	{
		public List<Neuron> Neurons { get; private set; }

		public Layer(int inputs, int neurons)
		{
			Random random = new Random();
			Neurons = new List<Neuron>();
			for (int i = 0; i < neurons; i++)
				Neurons.Add(new Neuron(inputs, random));
		}
	}
}

