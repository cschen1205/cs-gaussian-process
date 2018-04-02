using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    public class PredictorStat
    {
        //the mean absolute error (MAE) is a quantity used to measure how close forecasts or predictions are to the eventual outcomes
        //$MEA=\frac{\sum_{i=1}^n | f_i - y_i |}{n}$ where $f_i$ is the predition and $y_i$ is the true value
        private double mMAE;

        public static string DescMAE
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(@"The mean absolute error (MAE) is a quantity used to measure how close forecasts or predictions are to the eventual outcomes\r\n
        $MEA=\frac{\sum_{i=1}^n | f_i - y_i |}{n}$ where $f_i$ is the predition and $y_i$ is the true value");
                return sb.ToString();
            }
        }

        // the mean absolute scaled error (MASE) is a measure of the accuracy of forecasts
        // it is a "generally applicable measurement of forecast accuracy without the problems seen in the other measurements."
        //  $MASE = \frac{\sum_{t=1}^n (| \frac{e_t}{(1/(n-1)) * \sum_{i=2}^n | Y_i - Y_{i-1} | } |)}{n}  where $e_t = Y_t - F_t$, $F_t$ is the 
        // forecast value and $Y_t$ is the true value.
        private double mMASE;

        public static string DescMASE
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(@"the mean absolute scaled error (MASE) is a measure of the accuracy of forecasts
        it is a generally applicable measurement of forecast accuracy without the problems seen in the other measurements.\r\n
        $MASE = \frac{\sum_{t=1}^n (| \frac{e_t}{(1/(n-1)) * \sum_{i=2}^n | Y_i - Y_{i-1} | } |)}{n}  where $e_t = Y_t - F_t$, $F_t$ is the 
         forecast value and $Y_t$ is the true value.");
                return sb.ToString();
            }
        }

        // the mean squared error (MSE) of an estimator is one of many ways to quantify the difference between values implied 
        // by an estimator and the true values of the quantity being estimated
        private double mMSE;

        public static string DescMSE
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(@"the mean squared error (MSE) of an estimator is one of many ways to quantify the difference between values implied 
        by an estimator and the true values of the quantity being estimated");
                return sb.ToString();
            }
        }


        //the root mean squre error is the square root of MSE
        private double mRMSE;

        public static string DescRMSE
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(@"the root mean squre error is the square root of MSE");
                return sb.ToString();
            }
        }

        // the mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of 
        // accuracy of a method for constructing fitted time series values in statistics, specifically in trend estimation. It usually 
        // expresses accuracy as a percentage
        // $MAPE=(1 / n) * \sum_{t=1}^n | \frac{Y_t - F_t}{Y_t} |$ where $F_t$ is the predicted value and $Y_t$ is the actual value
        private double mMAPE;

        public static string DescMAPE
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(@"the mean absolute percentage error (MAPE), also known as mean absolute percentage deviation (MAPD), is a measure of 
        accuracy of a method for constructing fitted time series values in statistics, specifically in trend estimation. It usually 
        expresses accuracy as a percentage\r\n
        $MAPE=(1 / n) * \sum_{t=1}^n | \frac{Y_t - F_t}{Y_t} |$ where $F_t$ is the predicted value and $Y_t$ is the actual value\r\n
        Well-known drawback include divide-by-zero when actual value Y_t is zero and no restriction on upper level, alternatives that solve this include SMAPE");
                return sb.ToString();
            }
        }

        // Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors. 
        // SMAPE has advantage over MAPE as MAPE comparison result between different predictors will be distorted when comparing
        // across multiple time series.
        // $SMAPE= (1 / n) * \sum_{t=1}{n} \frac{| Y_t - F_t |}{(Y_t+F_t) / 2}$ where $F_t$ is the predicted value and $Y_t$ is the actual value
        private double mSMAPE;

        public static string DescSMAPE
        {
            get
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(@"Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors. 
        SMAPE has advantage over MAPE as MAPE comparison result between different predictors will be distorted when comparing
        across multiple time series.\r\n
        $SMAPE= (1 / n) * \sum_{t=1}{n} \frac{| Y_t - F_t |}{(Y_t+F_t) / 2}$ where $F_t$ is the predicted value and $Y_t$ is the actual value");
                return sb.ToString();
            }
        }

        public void Compute(List<List<double>> fValues, List<List<double>> yValues)
        {
            int rowCount = fValues.Count;
            int colCount = fValues[0].Count;
            double[] f = new double[rowCount * colCount];
            double[] y = new double[rowCount * colCount];

            for (int i = 0; i < rowCount; ++i)
            {
                for (int j = 0; j < colCount; ++j)
                {
                    f[i * colCount + j] = fValues[i][j];
                    y[i * colCount + j] = yValues[i][j];
                }
            }

            Compute(f, y);
        }

        // f: prediction
        // y: actual value
        public void Compute(double[] f, double[] y)
        {
            int n = f.Length;
            if (n == 0) return;

            double e_t = 0;
            double[] e = new double[n];
            double absolute_error_sum = 0;
            double sqr_error_sum = 0;
            double absolute_relative_error_sum = 0;
            double symmetric_error_sum = 0;
            double one_step_error_sum = 0;
            double ae_t = 0;
            for (int t = 0; t < n; ++t)
            {
                e_t = f[t] - y[t];
                e[t] = e_t;
                ae_t = System.Math.Abs(e_t);
                absolute_error_sum += ae_t;
                sqr_error_sum += e_t * e_t;

                symmetric_error_sum += (ae_t * 2 / (f[t] + y[t]));
                absolute_relative_error_sum += System.Math.Abs(e_t / y[t]);
                if (t > 0)
                {
                    one_step_error_sum += System.Math.Abs(y[t] - y[t - 1]);
                }
            }

            //compute mean absolute error
            mMAE = absolute_error_sum / n;

            //compute mean square error
            mMSE = sqr_error_sum / n;
            mRMSE = System.Math.Sqrt(mMSE);

            //compute mean absolute scale error
            mMASE = 0;
            for (int t = 0; t < n; ++t)
            {
                mMASE += (e[t] * (n - 1) / one_step_error_sum);
            }
            mMASE /= n;

            //compute mean absolute percentage error
            mMAPE = absolute_relative_error_sum / n;

            //compute symmetric mean absolute percentage error
            mSMAPE = symmetric_error_sum / n;

            mComputed = true;
        }

        private bool mComputed = false;

        public bool Computed
        {
            get { return mComputed; }
        }

        public double MAE
        {
            get { return mMAE; }
        }

        public double MASE
        {
            get { return mMASE; }
        }

        public double MSE
        {
            get { return mMSE; }
        }

        public double RMSE
        {
            get { return mRMSE; }
        }

        public double MAPE
        {
            get { return mMAPE; }
        }

        public double SMAPE
        {
            get { return mSMAPE; }
        }

        public PredictorStat Clone()
        {
            PredictorStat clone = new PredictorStat();
            clone.mSMAPE = mSMAPE;
            clone.mMAPE = mMAPE;
            clone.mRMSE = mRMSE;
            clone.mMSE = mMSE;
            clone.mMASE = mMASE;
            clone.mMAE = mMAE;

            return clone;
        }
    }
}
