using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using Encog.Neural.Networks;
    using Encog.Neural.Networks.Layers;
    using Encog.Engine.Network.Activation;
    using Encog.Neural.NeuralData;
    using Encog.Neural.Networks.Training;
    using Encog.Neural.Networks.Training.Propagation.Resilient;
    using Encog.Neural.Networks.Training.Propagation.Back;
    using Encog.Neural.Networks.Training.Propagation.Manhattan;
    using Encog.Neural.Networks.Training.Lma;

    using Encog.Neural.Data.Basic;

    using Encog.ML.Data;
    using Encog.ML.Data.Basic;
    using Encog.ML.Data.Temporal;

    using ABMath.ModelFramework.Data;
    public class MLPPredictor : Predictor
    {
        private int mNumberHiddenLayer = 1;
        private double mMaxError = 0.001;
        private int mMaxEpoch = 5000;

        public override string Type
        {
            get
            {
                return "MLP";
            }
        }


        public override string ToString()
        {
            return string.Format("MLP: Hidden_Layers={0}, MaxTrainingError={1}, MaxEpoch={2}, Method={3}, WindowSize={4}", NumberHiddenLayer, MaxError, MaxEpoch, Method, WindowSize);
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

        private BasicNetwork BuildModel(TimeSeries simulatedData, out double scale_factor)
        {
            double[][] inputs = new double[simulatedData.Count][];

            scale_factor = 0;
            for (int i = 0; i < simulatedData.Count; ++i)
            {
                if (scale_factor < simulatedData[i])
                {
                    scale_factor = simulatedData[i];
                }
            }

            for (int i = 0; i < simulatedData.Count; ++i)
            {
                inputs[i] = new double[mWindowSize];
                for (int j = 0; j < mWindowSize; ++j)
                {
                    int index = i - (mWindowSize - j);
                    if (index >= 0)
                    {
                        inputs[i][j] = simulatedData[index] / scale_factor;
                    }
                }
            }

            double[][] outputs = new double[simulatedData.Count][];
            for (int i = 0; i < simulatedData.Count; ++i)
            {
                outputs[i] = new double[1] { simulatedData[i] / scale_factor };
            }

            INeuralDataSet trainingSet = new BasicNeuralDataSet(inputs, outputs);

            BasicNetwork network = new BasicNetwork();

            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, mWindowSize));
            for (int i = 0; i < mNumberHiddenLayer; ++i)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, mWindowSize));
            }
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            Train(network, trainingSet);

            return network;
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

        private void Train(BasicNetwork network, IMLDataSet trainingSet)
        {
            if (mTrainingMethod == TrainingMethod.ResilientPropagation)
            {
                ITrain train = new ResilientPropagation(network, trainingSet);

                int epoch = 1;
                do
                {
                    train.Iteration();
                    epoch++;
                } while (train.Error > mMaxError && epoch < mMaxEpoch);
            }
            else if (mTrainingMethod == TrainingMethod.LevenbergMarquardt)
            {
                LevenbergMarquardtTraining train = new LevenbergMarquardtTraining(network, trainingSet);

                int epoch = 1;
                do
                {
                    train.Iteration();
                    epoch++;
                } while (train.Error > mMaxError && epoch < mMaxEpoch);
            }
            else if (mTrainingMethod == TrainingMethod.Backpropagation)
            {
                Backpropagation train = new Backpropagation(network, trainingSet);

                int epoch = 1;
                do
                {
                    train.Iteration();
                    epoch++;
                } while (train.Error > mMaxError && epoch < mMaxEpoch);
            }
            else if (mTrainingMethod == TrainingMethod.ManhattanPropagation)
            {
                ManhattanPropagation train = new ManhattanPropagation(network, trainingSet, 0.9);
                int epoch = 1;
                do
                {
                    train.Iteration();
                    epoch++;
                } while (train.Error > mMaxError && epoch < mMaxEpoch);
            }
        }

        private double mScaleFactor;
        private BasicNetwork mModel = null;
        public override TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            if (mModel == null)
            {
                mModel = BuildModel(simulatedData, out mScaleFactor);
            }

            //TimeSeries preds = new TimeSeries();

            //for (int i = 0; i < futureTimes.Count; ++i)
            //{
            //    double[] input = new double[mWindowSize];
            //    for (int j = 0; j < mWindowSize; ++j)
            //    {
            //        int index = i - (mWindowSize - 1 - j);
            //        if (index >= 0)
            //        {
            //            input[j] = simulatedData[index] / mScaleFactor;
            //        }
            //    }

            //    double[] output = new double[1];
            //    mModel.Compute(input, output);
            //    preds.Add(futureTimes[i], output[0] * mScaleFactor, false);
            //}

            return Forecast(mModel, mScaleFactor, simulatedData, futureTimes);
        }

        public TimeSeries Forecast(BasicNetwork network, double scale_factor, TimeSeries simulatedData, List<DateTime> futureTimes)
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
                        input_val = data[idx2] / scale_factor;
                    }
                    input[i] = input_val;
                }
                IMLData output = network.Compute(input);
                double prediction = output[0] * scale_factor;
                data[data_count + idx] = prediction;
                ts.Add(futureTimes[idx], prediction, false);
            }

            return ts;
        }

        public override void ResetModel()
        {
            mModel = null;
        }

        protected override TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            if (mModel == null)
            {
                mModel = BuildModel(simulatedData, out mScaleFactor);
            }

            TimeSeries preds = new TimeSeries();

            for (int i = 0; i < simulatedData.Count; ++i)
            {
                double[] input = new double[mWindowSize];
                for (int j = 0; j < mWindowSize; ++j)
                {
                    int index = i - (mWindowSize - j);
                    if (index >= 0)
                    {
                        input[j] = simulatedData[index] / mScaleFactor;
                    }
                }

                double[] output = new double[1];
                mModel.Compute(input, output);
                preds.Add(simulatedData.TimeStamp(i), output[0] * mScaleFactor, false);
            }

            return preds;
        }

        public override Predictor Clone()
        {
            MLPPredictor p = new MLPPredictor();
            p.MaxEpoch = mMaxEpoch;
            p.MaxError = mMaxError;
            p.NumberHiddenLayer = mNumberHiddenLayer;
            p.Method = mTrainingMethod;
            p.WindowSize = mWindowSize;
            return p;
        }
    }
}
