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

    public class ARMAXPredictor : Predictor
    {
        private int mAROrder = 4; //auto regression order
        private int mMAOrder = 3; //moving average order
        private int mNumberIterLDS = 200; //number of discrepancy sequence iterations, should be at least about 200
        private int mNumberIterOpt = 100; //number of standard optimizer iterations, should be at least about 100

        public override string Type
        {
            get
            {
                return "ARMAX";
            }
        }

        public override TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            // fit model first, using maximum likelihood estimation
            var model = new ARMAModel(mAROrder, mMAOrder);     // create the model object

            model.TheData = simulatedData;       // this is the data we want to fit the model to
            model.FitByMLE(mNumberIterLDS, mNumberIterOpt, 0, null);   // first param is # low discrepancy sequence iterations, should be at least about 200
            // second param is # standard optimizer iterations, should be at least about 100

            // now do some forecasting beyond the end of the data
            var forecaster = new ForecastTransform();

            forecaster.FutureTimes = futureTimes.ToArray(); // these are future times at which we want forecasts
            // for now, ARMA models do not use the times in any meaningful way when forecasting,
            // they just assume that these are the timestamps for the next sequential points
            // after the end of the existing time series

            forecaster.SetInput(0, model, null);            // the ARMA model used for forecasting
            forecaster.SetInput(1, simulatedData, null);    // the original data

            // normally you would call the Recompute() method of a transform, but there is no need
            // here; it is automatically called as soon as all inputs are set to valid values (in the 2 statements above)
            var predictors = forecaster.GetOutput(0) as TimeSeries;

            return predictors;
        }

        protected override TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            ARMAModel model = new ARMAModel(mAROrder, mMAOrder);

            model.SetInput(0, simulatedData, null);
            //Maximum Likelihood Estimation
            model.FitByMLE(mNumberIterLDS, mNumberIterOpt, 0, null);   // first param is # low discrepancy sequence iterations, should be at least about 200
            // second param is # standard optimizer iterations, should be at least about 100

            //Compute the residuals
            model.ComputeResidualsAndOutputs();

            //model1.GetOutputName(3);
            //model1.Description;

            //Get the predicted values
            return model.GetOutput(3) as TimeSeries;
        }

        public override string ToString()
        {
            return "ARMAX";
        }

        public int AROrder
        {
            get { return mAROrder; }
            set { mAROrder = value; }
        }

        public int MAOrder
        {
            get { return mMAOrder; }
            set { mMAOrder = value; }
        }

        public int LDSIters
        {
            get { return mNumberIterLDS; }
            set { mNumberIterLDS = value; }
        }

        public int OptIters
        {
            get { return mNumberIterOpt; }
            set { mNumberIterOpt = value; }
        }

        public override Predictor Clone()
        {
            ARMAXPredictor p = new ARMAXPredictor();
            p.MAOrder = mMAOrder;
            p.AROrder = mAROrder;
            p.LDSIters = LDSIters;
            p.OptIters = OptIters;
            p.WindowSize = mWindowSize;

            return p;
        }
    }
}
