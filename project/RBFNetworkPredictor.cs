using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using ABMath.ModelFramework.Data;

    using Encog.Util;
    using Encog.Util.Arrayutil;

    using Encog.Engine.Network.Activation;

    using Encog.Neural.Networks;
    using Encog.Neural.RBF;
    using Encog.Neural.Networks.Layers;
    using Encog.Neural.Networks.Training;
    using Encog.Neural.Networks.Training.Propagation.Resilient;
    using Encog.Neural.Networks.Training.Propagation.Back;
    using Encog.Neural.Networks.Training.Propagation.Manhattan;
    using Encog.Neural.Networks.Training.Lma;

    using Encog.MathUtil.RBF;
    using Encog.Neural.Rbf.Training;
    using Encog.Util.Simple;
    using Encog.Neural.Pattern;

    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Temporal;

    public class RBFNetworkPredictor : Predictor
    {
        private double mMaxError = 0.001;
        protected int mNumberHiddenLayer = 1;
        private int mMaxEpoch = 5000;
        private double mNormalizedLow = 0;
        private double mNormalzedHigh = 1;

        public override string Type
        {
            get
            {
                return "RBF";
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

        public double MaxError
        {
            get { return mMaxError; }
            set
            {
                if (mMaxError != value)
                {
                    mMaxError = value;
                    ResetModel();
                }
            }
        }

        public int NumberHiddenLayer
        {
            get { return mNumberHiddenLayer; }
            set
            {
                if (mNumberHiddenLayer != value)
                {
                    mNumberHiddenLayer = value;
                    ResetModel();
                }
            }
        }

        public int MaxEpoch
        {
            get { return mMaxEpoch; }
            set
            {
                if (mMaxEpoch != value)
                {
                    mMaxEpoch = value;
                    ResetModel();
                }
            }
        }

        public override string ToString()
        {
            return string.Format("RBF: MaxTrainingError={0}, MaxEpoch={1}, Method=CVD, WindowSize={2}", MaxError, MaxEpoch, WindowSize);
        }

        public override void ResetModel()
        {
            mModel = null;
        }

        protected virtual RBFNetwork BuildNetwork(TimeSeries simulatedData, out NormalizeArray norm)
        {
            double[] data = GenerateData(simulatedData);
            double[] normalizedData = NormalizeData(data, mNormalizedLow, mNormalzedHigh, out norm);
            RBFNetwork network = CreateNetwork();
            IMLDataSet training = GenerateTraining(normalizedData);
            Train(network, training);

            return network;
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

        public double[] NormalizeData(double[] data, double lo, double hi, out NormalizeArray norm)
        {
            norm = new NormalizeArray { NormalizedLow = lo, NormalizedHigh = hi };
            return norm.Process(data);
        }

        public IMLDataSet GenerateTraining(double[] normalizedData)
        {
            var result = new TemporalMLDataSet(WindowSize, 1);

            TemporalDataDescription desc = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
            result.AddDescription(desc);

            int TrainStart = 0;
            int TrainEnd = normalizedData.Length;

            for (int index = TrainStart; index < TrainEnd; index++)
            {
                TemporalPoint point = new TemporalPoint(1) { Sequence = index };
                point.Data[0] = normalizedData[index];
                result.Points.Add(point);
            }

            result.Generate();
            return result;
        }

        public enum TrainingMethod
        {
            ResilientPropagation,
            LevenbergMarquardt,
            Backpropagation,
            ManhattanPropagation
        }

        private TrainingMethod mTrainingMethod = TrainingMethod.ResilientPropagation;

        public TrainingMethod Method
        {
            get
            {
                return mTrainingMethod;
            }
            set
            {
                if (mTrainingMethod != value)
                {
                    mTrainingMethod = value;
                    ResetModel();
                }
            }
        }

        private void Train(RBFNetwork network, IMLDataSet trainingSet)
        {
            var train = new SVDTraining(network, trainingSet);

            //SVD is a single step solve
            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error:" + train.Error);
                epoch++;
            } while ((epoch < 1) && (train.Error > mMaxError));
        }

        private bool includeEdgeRBFs = true;
        private int numNeuronsPerDimension = 7;

        protected virtual RBFNetwork CreateNetwork()
        {
            //RBFNetwork network = new RBFNetwork(WindowSize, 1, new IRadialBasisFunction[]{new GaussianFunction()});

            //General setup is the same as before
            double volumeNeuronWidth = 2.0 / numNeuronsPerDimension;

            var pattern = new RadialBasisPattern();
            pattern.InputNeurons = WindowSize;
            pattern.OutputNeurons = 1;

            //Total number of neurons required.
            //Total number of Edges is calculated possibly for future use but not used any further here
            int numNeurons = (int)System.Math.Pow(numNeuronsPerDimension, WindowSize);
            // int numEdges = (int) (dimensions*Math.Pow(2, dimensions - 1));

            pattern.AddHiddenLayer(numNeurons);

            var network = (RBFNetwork)pattern.Generate();

            //Position the multidimensional RBF neurons, with equal spacing, within the provided sample space from 0 to 1.
            network.SetRBFCentersAndWidthsEqualSpacing(0, 1, RBFEnum.Gaussian, volumeNeuronWidth, includeEdgeRBFs);

            return network;
        }

        private NormalizeArray mNorm;
        private RBFNetwork mModel = null;
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

        public TimeSeries Predict(RBFNetwork network, NormalizeArray norm, TimeSeries simulatedData)
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

        public TimeSeries Forecast(RBFNetwork network, NormalizeArray norm, TimeSeries simulatedData, List<DateTime> futureTimes)
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
            RBFNetworkPredictor p = new RBFNetworkPredictor();
            p.NormalizedHigh = mNormalzedHigh;
            p.NormalizedLow = mNormalizedLow;
            p.MaxEpoch = mMaxEpoch;
            p.MaxError = mMaxError;
            p.NumberHiddenLayer = mNumberHiddenLayer;
            p.Method = mTrainingMethod;
            p.WindowSize = mWindowSize;
            return p;
        }
    }
}
