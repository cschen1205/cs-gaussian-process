using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using AForge;
    using AForge.Genetic;
    using ABMath.ModelFramework.Data;
    public class GeneticProgrammingPredictor : Predictor
    {
        private int mPopSize = 100;
        private int mMaxEpoch = 1;

        public int MaxEpoch
        {
            get { return mMaxEpoch; }
            set { mMaxEpoch = value; }
        }

        public int PopSize
        {
            get { return mPopSize; }
            set { mPopSize = value; }
        }

        public override string Type
        {
            get
            {
                return "Genetic Programming";
            }
        }

        public override TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            // time series to predict
            double[] data = GetDataArray(simulatedData);

            // constants
            double[] constants = new double[10] { 1, 2, 3, 5, 7, 11, 13, 17, 19, 23 };

            TimeSeriesPredictionFitness tsp_fitness = new TimeSeriesPredictionFitness(data, mWindowSize, 1, constants);


            // create population
            Population population = new Population(mPopSize,
            new GPTreeChromosome(new SimpleGeneFunction(mWindowSize + constants.Length)),
            tsp_fitness,
            new EliteSelection());

            for (int i = 0; i < mMaxEpoch; ++i)
            {
                // run one epoch of the population
                population.RunEpoch();
            }

            GPTreeChromosome best_solution = (GPTreeChromosome)population.BestChromosome;

            int tdata_count = simulatedData.Count;
            double[] all_data = new double[tdata_count + futureTimes.Count + 1];

            for (int i = 0; i < all_data.Length; ++i)
            {
                all_data[i] = 0;
            }


            for (int i = 0; i < tdata_count; ++i)
            {
                all_data[i] = simulatedData[i];
            }

            double[] predicted_data;
            Predict(best_solution, all_data, constants, mWindowSize, 1, out predicted_data);

            TimeSeries ts = new TimeSeries();
            for (int i = 0; i < futureTimes.Count; ++i)
            {
                ts.Add(futureTimes[i], predicted_data[i + tdata_count], true);
            }

            return ts;
        }

        public double[] GetDataArray(TimeSeries ts)
        {
            double[] data = new double[ts.Count];

            for (int i = 0; i < ts.Count; ++i)
            {
                data[i] = ts[i];
            }
            return data;
        }

        protected override TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            // time series to predict
            double[] data = GetDataArray(simulatedData);

            // constants
            double[] constants = new double[10] { 1, 2, 3, 5, 7, 11, 13, 17, 19, 23 };

            TimeSeriesPredictionFitness tsp_fitness = new TimeSeriesPredictionFitness(data, mWindowSize, 1, constants);


            // create population
            Population population = new Population(mPopSize,
            new GPTreeChromosome(new SimpleGeneFunction(mWindowSize + constants.Length)),
            tsp_fitness,
            new EliteSelection());

            for (int i = 0; i < mMaxEpoch; ++i)
            {
                // run one epoch of the population
                population.RunEpoch();
            }

            GPTreeChromosome best_solution = (GPTreeChromosome)population.BestChromosome;

            double[] predicted_data;
            Predict(best_solution, data, constants, mWindowSize, 1, out predicted_data);

            TimeSeries ts = new TimeSeries();
            for (int i = 0; i < data.Length; ++i)
            {
                ts.Add(simulatedData.TimeStamp(i), predicted_data[i], true);
            }

            return ts;
        }

        public static void Predict(IChromosome chromosome, double[] data, double[] constants, int windowSize, int predictionSize, out double[] predicted_data)
        {
            predicted_data = new double[data.Length];

            for (int i = 0; i < data.Length; ++i)
            {
                predicted_data[i] = data[i];
            }

            // get function in polish notation
            string function = chromosome.ToString();

            double[] variables = new double[constants.Length + windowSize];

            // go through all the data
            for (int i = 0, n = data.Length - windowSize - predictionSize; i < n; i++)
            {
                // put values from current window as variables
                for (int j = 0, b = i + windowSize - 1; j < windowSize; j++)
                {
                    variables[j] = data[b - j];
                }

                // avoid evaluation errors
                try
                {
                    // evaluate the function
                    double y = PolishExpression.Evaluate(function, variables);
                    // check for correct numeric value
                    if (double.IsNaN(y))
                    {
                        y = 0;
                    }

                    predicted_data[i + windowSize] = y;
                }
                catch
                {
                    predicted_data[i + windowSize] = 0;
                }
            }
        }

        public override Predictor Clone()
        {
            GeneticProgrammingPredictor clone = new GeneticProgrammingPredictor();
            clone.WindowSize = mWindowSize;
            clone.PopSize = mPopSize;
            clone.MaxEpoch = mMaxEpoch;

            return clone;
        }
    }
}
