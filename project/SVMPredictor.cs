using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.SVM;
    using Encog.ML.SVM.Training;
    using Encog.ML.Train;
    using Encog.ML.Train.Strategy;
    using Encog.Neural.Networks.Training;
    using Encog.Neural.Networks.Training.Anneal;
    using Encog.Neural.Networks.Training.Propagation.Back;
    using Encog.Util;
    using Encog.Util.Arrayutil;

    using ABMath.ModelFramework.Data;

    public class SVMPredictor : Predictor
    {
        private double mNormalizedLow = 0;
        private double mNormalzedHigh = 1;

        public override string Type
        {
            get
            {
                return "SVM";
            }
        }

        public double NormalizedHigh
        {
            get { return mNormalzedHigh; }
            set
            {
                if (mNormalzedHigh != value)
                {
                    mNormalzedHigh = value;
                    ResetModel();
                }
            }
        }

        public double NormalizedLow
        {
            get { return mNormalizedLow; }
            set
            {
                if (mNormalizedLow != value)
                {
                    mNormalizedLow = value;
                    ResetModel();
                }
            }
        }

        public double[] NormalizeData(double[] data, double lo, double hi, out NormalizeArray norm)
        {
            norm = new NormalizeArray();
            norm.NormalizedHigh = (hi);
            norm.NormalizedLow = lo;

            // create arrays to hold the normalized sunspots
            double[] normalizedData = norm.Process(data);


            return normalizedData;
        }

        private static SupportVectorMachine networkx = new SupportVectorMachine();

        public IMLDataSet GenerateTraining(double[] normalizedData)
        {
            TemporalWindowArray temp = new TemporalWindowArray(mWindowSize, 1);
            temp.Analyze(normalizedData);
            return temp.Process(normalizedData);
        }

        public SupportVectorMachine CreateNetwork()
        {
            SupportVectorMachine network = new SupportVectorMachine(mWindowSize, true);
            return network;
        }

        public void train(SupportVectorMachine network, IMLDataSet training)
        {
            SVMTrain train = new SVMTrain(network, training);
            train.Iteration();
        }

        public SupportVectorMachine SVMSearch(SupportVectorMachine anetwork, IMLDataSet training)
        {
            SVMSearchTrain bestsearch = new SVMSearchTrain(anetwork, training);
            StopTrainingStrategy stop = new StopTrainingStrategy(0.00000000001, 1);
            bestsearch.AddStrategy(stop);
            while (bestsearch.IterationNumber < 30 && !stop.ShouldStop())
            {
                bestsearch.Iteration();
                Console.WriteLine("Iteration #" + bestsearch.IterationNumber + " Error :" + bestsearch.Error);
            }

            bestsearch.FinishTraining();

            return anetwork;
        }

        public double[] GenerateData(TimeSeries simulatedData)
        {
            int data_count = simulatedData.Count;
            double[] data = new double[data_count];
            for (int index = 0; index < data_count; ++index)
            {
                double data_val = simulatedData[index];
                data[index] = data_val;
            }

            return data;
        }

        public SupportVectorMachine BuildNetwork(TimeSeries simulatedData, out NormalizeArray norm)
        {
            double[] data = GenerateData(simulatedData);
            double[] normalizedData = NormalizeData(data, 0.1, 0.9, out norm);

            SupportVectorMachine network = CreateNetwork();
            IMLDataSet training = GenerateTraining(normalizedData);
            SupportVectorMachine trained = SVMSearch(network, training);

            train(trained, training);

            return trained;
        }

        public override void ResetModel()
        {
            mModel = null;
        }

        private NormalizeArray mNorm;
        private SupportVectorMachine mModel = null;
        public override TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            if (mModel == null)
            {
                mModel = BuildNetwork(simulatedData, out mNorm);
            }
            return Forecast(mModel, mNorm, simulatedData, futureTimes);
        }

        protected override TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            if (mModel == null)
            {
                mModel = BuildNetwork(simulatedData, out mNorm);
            }
            return Predict(mModel, mNorm, simulatedData);
        }

        public TimeSeries Predict(SupportVectorMachine network, NormalizeArray norm, TimeSeries simulatedData)
        {
            double[] data = GenerateData(simulatedData);

            int data_count = simulatedData.Count;
            TimeSeries ts = new TimeSeries();
            double input_val = 0;
            for (int idx = 0; idx < data_count; ++idx)
            {
                var input = new BasicMLData(WindowSize);
                for (var i = 0; i < WindowSize; i++)
                {
                    int idx2 = (idx - WindowSize) + i;
                    if (idx2 < 0)
                    {
                        input_val = 0;
                    }
                    else
                    {
                        input_val = norm.Stats.Normalize(data[idx2]);
                    }
                    input[i] = input_val;
                }
                IMLData output = network.Compute(input);
                double prediction = norm.Stats.DeNormalize(output[0]);
                ts.Add(simulatedData.TimeStamp(idx), prediction, false);
            }

            return ts;
        }

        public TimeSeries Forecast(SupportVectorMachine network, NormalizeArray norm, TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            int data_count = simulatedData.Count;
            int future_data_count = futureTimes.Count;

            double[] data = new double[data_count + future_data_count];

            for (int idx = 0; idx < data_count; ++idx)
            {
                data[idx] = simulatedData[idx];
            }
            for (int idx = 0; idx < future_data_count; ++idx)
            {
                data[data_count + idx] = 0;
            }

            TimeSeries ts = new TimeSeries();
            double input_val = 0;
            for (int idx = 0; idx < future_data_count; ++idx)
            {
                var input = new BasicMLData(WindowSize);
                for (var i = 0; i < WindowSize; i++)
                {
                    int idx2 = (data_count + idx - WindowSize) + i;
                    if (idx2 < 0)
                    {
                        input_val = 0;
                    }
                    else
                    {
                        input_val = norm.Stats.Normalize(data[idx2]);
                    }
                    input[i] = input_val;
                }
                IMLData output = network.Compute(input);
                double prediction = norm.Stats.DeNormalize(output[0]);
                data[data_count + idx] = prediction;
                ts.Add(futureTimes[idx], prediction, false);
            }

            return ts;
        }

        public override Predictor Clone()
        {
            SVMPredictor p = new SVMPredictor();
            p.NormalizedHigh = mNormalzedHigh;
            p.NormalizedLow = mNormalizedLow;
            p.WindowSize = mWindowSize;
            return p;
        }
    }
}
