using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using SimuKit.Math.Distribution;
    using ABMath.ModelFramework.Data;
    public class GaussianDistributionPredictor : Predictor
    {
        private bool mPositiveValueOnly = true;
        public bool PositveValueOnly
        {
            get { return mPositiveValueOnly; }
            set
            {
                if (mPositiveValueOnly != value)
                {
                    mPositiveValueOnly = value;
                    ResetModel();
                }
            }
        }

        public override string Type
        {
            get
            {
                return "Gaussian Distribution";
            }
        }

        public Gaussian BuildModel(TimeSeries simulatedData)
        {
            int data_count = simulatedData.Count;
            double[] data = new double[data_count];
            for (int i = 0; i < data_count; ++i)
            {
                data[i] = simulatedData[i];
            }
            Gaussian model = new Gaussian();
            model.Process(data);
            return model;
        }

        public override void ResetModel()
        {
            mModel = null;
        }

        public override string ToString()
        {
            double mean = 0;
            double stdDev = 0;
            if (mModel != null)
            {
                mean = mModel.Mean;
                stdDev = mModel.StdDev;
            }
            return string.Format("Gaussian Distribution: Mean={0}, StdDev={1}", mean, stdDev);
        }

        private Gaussian mModel = null;
        public override TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            if (mModel == null)
            {
                mModel = BuildModel(simulatedData);
            }

            TimeSeries ts = new TimeSeries();
            int future_data_count = futureTimes.Count;
            for (int i = 0; i < future_data_count; ++i)
            {
                double prediction = mModel.Next();
                if (mPositiveValueOnly)
                {
                    prediction = System.Math.Max(0, prediction);
                }
                ts.Add(futureTimes[i], prediction, false);
            }
            return ts;
        }

        protected override TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            if (mModel == null)
            {
                mModel = BuildModel(simulatedData);
            }


            TimeSeries ts = new TimeSeries();
            int data_count = simulatedData.Count;
            for (int i = 0; i < data_count; ++i)
            {
                double prediction = mModel.Next();
                if (mPositiveValueOnly)
                {
                    prediction = System.Math.Max(0, prediction);
                }
                ts.Add(simulatedData.TimeStamp(i), prediction, false);
            }
            return ts;
        }

        public override Predictor Clone()
        {
            GaussianDistributionPredictor p = new GaussianDistributionPredictor();
            p.WindowSize = mWindowSize;
            return p;
        }
    }
}
