using System;

namespace CWIANN
{
	public class DataSet
	{
		public string Label { get; }
		public double Data { get; }
		public double Target { get; }

		public DataSet (string pathFile, int indexTarget)
		{
			LoadData(pathFile);
			LoadTarget(pathFile, indexTarget);
		}

		private void LoadData(string pathFile)
		{

		}

		private void LoadTarget(string pathFile, int indexTarget)
		{

		}

	}
}

