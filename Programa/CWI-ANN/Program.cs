using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using AForge.Neuro.Learning;
using AForge.Neuro;

namespace CWIANN
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			// Load my Model
			DateTime start = DateTime.Now;
			Console.WriteLine("Starting my ANN Multilayer Perceptron...");
			MultilayerPerceptron mlp = new MultilayerPerceptron(3, 4, 1);
			mlp.Training(
				new double[,] { { 1, 1, 1 }, { 1, 0, 1 }, { 0, 0, 0 }, { 0, 1, 1 }, { 1, 0, 0 } }, 
				new double[] { 1, 1, 0, 1, 1 }
			);
			Console.WriteLine("Result: " + Math.Round(mlp.Run(new double[] { 0, 1, 0 }).FirstOrDefault(), 1));
			Console.WriteLine("Processing time: {0} seconds", (DateTime.Now - start).Seconds);
			Console.WriteLine("-----------------");


			// Load AForge Model
			Console.WriteLine("Starting AForge ANN Multilayer Perceptron...");

			// initialize input and output values
			double[][] input = new double[4][] {
				new double[] {0, 0}, new double[] {0, 1},
				new double[] {1, 0}, new double[] {1, 1}
			};

			double[][] output = new double[4][] {
				new double[] {0}, new double[] {1},
				new double[] {1}, new double[] {0}
			};

			ActivationNetwork network = new ActivationNetwork(
				new SigmoidFunction(2),
				2, // two inputs in the network
				2, // two neurons in the first layer
				1); // one neuron in the second layer
			
			// create teacher
			BackPropagationLearning teacher = new BackPropagationLearning(network);
			int iteration = 0;
			while (true)
			{
				iteration++;
				double error = teacher.RunEpoch(input, output);
				System.Diagnostics.Debug.WriteLine(iteration);
				if (error < 0.001) break;
			}
			Console.WriteLine("Iterations: " + iteration + " to XOR");

			while (true)
			{
				Console.WriteLine("Primeiro Valor: (0 ou 1):");
				double f1 = (double)Convert.ToInt32(Console.ReadLine());
				Console.WriteLine("Segundo Valor: (0 ou 1):");
				double f2 = (double)Convert.ToInt32(Console.ReadLine());
				double[] netout = network.Compute(new double[2] { f1, f2 });
				Console.WriteLine("{0} XOR {1} = {2}.", (int)f1, (int)f2, netout[0]);
				Console.ReadLine();
			}
		}
	}
}
