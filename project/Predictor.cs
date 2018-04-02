using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using ABMath.ModelFramework.Models;
    using ABMath.ModelFramework.Data;
    using ABMath.ModelFramework.Transforms;
    using MathNet.Numerics.Distributions;
    //using MathNet.Numerics.RandomSources;
    using MathNet.Numerics;

    public abstract class Predictor
    {
        public delegate void OutputBuiltHandler(Predictor p);
        public event OutputBuiltHandler OutputBuilt;

        protected int mWindowSize = 4;
        public int WindowSize
        {
            get { return mWindowSize; }
            set
            {
                if (mWindowSize != value)
                {
                    mWindowSize = value;
                    ResetModel();
                }
            }
        }

        public virtual void ResetModel()
        {

        }

        public TimeSeries BuildOutput(DateTime start_time, List<int> data, int used_data_count, int day_interval, object userState = null)
        {
            //Make new time series variable
            TimeSeries simulatedData = new TimeSeries();
            DateTime current = start_time;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                int s1 = data[t];
                simulatedData.Add(current, s1, false);
                current = new DateTime(current.AddDays(day_interval).Ticks);
            }
            return BuildOutput(simulatedData, userState);
        }

        public virtual Predictor Clone()
        {
            return null;
        }

        public TimeSeries BuildOutput(DateTime start_time, List<double> data, int used_data_count, int day_interval, object userState = null)
        {
            //Make new time series variable
            TimeSeries simulatedData = new TimeSeries();
            DateTime current = start_time;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                double s1 = data[t];
                simulatedData.Add(current, s1, false);
                current = new DateTime(current.AddDays(day_interval).Ticks);
            }
            return BuildOutput(simulatedData, userState);
        }

        public List<double> BuildOutput(List<double> data)
        {
            List<double> output = new List<double>();
            TimeSeries ts = BuildOutput(DateTime.Now, data, data.Count, 1);
            for (int t = 0; t < ts.Count; ++t)
            {
                output.Add(ts[t]);
            }
            return output;
        }

        public double[] BuildOutput(double[] data)
        {
            TimeSeries ts = BuildOutput(DateTime.Now, data.ToList(), data.Length, 1);
            double[] output = new double[ts.Count];
            for (int t = 0; t < ts.Count; ++t)
            {
                output[t] = ts[t];
            }

            return output;
        }

        public TimeSeries BuildOutput(List<DateTime> x_values, List<double> y_values, object userState = null)
        {
            //Make new time series variable
            TimeSeries simulatedData = new TimeSeries();
            int used_data_count = x_values.Count;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                double s1 = y_values[t];
                DateTime current = x_values[t];
                simulatedData.Add(current, s1, false);
            }
            return BuildOutput(simulatedData, userState);
        }

        public TimeSeries BuildOutput(List<int> x_values, List<double> y_values, object userState = null)
        {
            TimeSeries simulatedData = new TimeSeries();
            int used_data_count = x_values.Count;
            DateTime start = DateTime.UtcNow;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                double s1 = y_values[t];
                DateTime current = start.AddDays(x_values[t]);
                simulatedData.Add(current, s1, false);
            }
            return BuildOutput(simulatedData, userState);
        }

        private PredictorStat mStat = new PredictorStat();

        public PredictorStat Stat
        {
            get { return mStat; }
        }

        public TimeSeries BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            TimeSeries ts = _BuildOutput(simulatedData, userState);

            int data_count = simulatedData.Count;
            double[] f = new double[data_count];
            double[] y = new double[data_count];

            for (int idx = 0; idx < data_count; ++idx)
            {
                f[idx] = ts[idx];
                y[idx] = simulatedData[idx];
            }

            mStat.Compute(f, y);

            if (OutputBuilt != null)
            {
                OutputBuilt(this);
            }

            return ts;
        }

        protected abstract TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null);

        public TimeSeries BuildForecasts(List<DateTime> x_values, List<double> y_values, int day_interval, int future_count)
        {
            TimeSeries simulatedData = new TimeSeries();
            int used_data_count = x_values.Count;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                double s1 = y_values[t];
                DateTime current = x_values[t];
                simulatedData.Add(current, s1, false);
            }


            var nextTime = simulatedData.GetLastTime();
            var futureTimes = new List<DateTime>();
            for (int t = 0; t < future_count; ++t)                    // go eight days into the future beyond the end of the data we have
            {
                nextTime = nextTime.AddDays(day_interval);
                futureTimes.Add(nextTime);
            }

            return BuildForecasts(simulatedData, futureTimes);
        }

        public List<double> BuildForecasts(List<double> data, int future_count)
        {
            TimeSeries simulatedData = new TimeSeries();
            DateTime current = DateTime.UtcNow;
            //Create the data
            for (int t = 0; t < data.Count; ++t)
            {
                double s1 = data[t];
                simulatedData.Add(current, s1, false);
                current = current.AddDays(1);
            }

            var nextTime = simulatedData.GetLastTime();
            var futureTimes = new List<DateTime>();
            for (int t = 0; t < future_count; ++t)                    // go eight days into the future beyond the end of the data we have
            {
                nextTime = nextTime.AddDays(1);
                futureTimes.Add(nextTime);
            }

            TimeSeries preds1 = BuildForecasts(simulatedData, futureTimes);
            List<double> preds2 = new List<double>();
            for (int i = 0; i < future_count; ++i)
            {
                preds2.Add(preds1[i]);
            }

            return preds2;
        }

        public TimeSeries BuildForecasts(DateTime start_time, List<int> data, int used_data_count, int day_interval, int future_count)
        {
            TimeSeries simulatedData = new TimeSeries();
            DateTime current = start_time;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                double s1 = data[t];
                simulatedData.Add(current, s1, false);
                current = new DateTime(current.AddDays(day_interval).Ticks);
            }

            var nextTime = simulatedData.GetLastTime();
            var futureTimes = new List<DateTime>();
            for (int t = 0; t < future_count; ++t)                    // go eight days into the future beyond the end of the data we have
            {
                nextTime = nextTime.AddDays(day_interval);
                futureTimes.Add(nextTime);
            }

            return BuildForecasts(simulatedData, futureTimes);
        }

        public TimeSeries BuildForecasts(DateTime start_time, List<double> data, int used_data_count, int day_interval, int future_count)
        {
            TimeSeries simulatedData = new TimeSeries();
            DateTime current = start_time;
            //Create the data
            for (int t = 0; t < used_data_count; ++t)
            {
                double s1 = data[t];
                simulatedData.Add(current, s1, false);
                current = new DateTime(current.AddDays(day_interval).Ticks);
            }

            var nextTime = simulatedData.GetLastTime();
            var futureTimes = new List<DateTime>();
            for (int t = 0; t < future_count; ++t)                    // go eight days into the future beyond the end of the data we have
            {
                nextTime = nextTime.AddDays(day_interval);
                futureTimes.Add(nextTime);
            }

            return BuildForecasts(simulatedData, futureTimes);
        }

        public abstract TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes);

        protected string mName = "";
        public string Name
        {
            get { return mName; }
            set { mName = value; }
        }

        public virtual string Type
        {
            get
            {
                return "";
            }
        }
    }
}
