using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.TimeSeries
{
    using ABMath.ModelFramework.Data;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Double.Factorization;

    public class GaussianProcessPredictor : Predictor
    {
        public enum BuildOutputType
        {
            Mean,
            StdDev
        }

        private BuildOutputType mBuildOutputType = BuildOutputType.Mean;
        public BuildOutputType OutputType
        {
            get { return mBuildOutputType; }
            set { mBuildOutputType = value; }
        }

        public override string Type
        {
            get
            {
                return "Gaussian Process";
            }
        }

        //amplitude
        private double mSigma0 = 1;
        public double Sigma0
        {
            get
            {
                return mSigma0;
            }
            set
            {
                mSigma0 = value;
            }
        }

        private double mSigmaN = 0.1;
        public double SigmaN
        {
            get
            {
                return mSigmaN;
            }
            set
            {
                mSigmaN = value;
            }
        }

        //length scale
        private double mLambda = 1;
        public double Lambda
        {
            get
            {
                return mLambda;
            }
            set
            {
                mLambda = value;
            }
        }

        public override TimeSeries BuildForecasts(TimeSeries simulatedData, List<DateTime> futureTimes)
        {
            int future_point_count = futureTimes.Count;

            double[] targets = GetDataArray(simulatedData);

            double[] fstar;
            double[] Vfsar;
            double logpyX;
            Predict(targets, future_point_count, mSigma0, mLambda, mSigmaN, out fstar, out Vfsar, out logpyX);

            TimeSeries ft = new TimeSeries();
            for (int i = 0; i < future_point_count; ++i)
            {
                ft.Add(futureTimes[i], fstar[targets.Length + i], true);
            }
            return ft;
        }

        protected override TimeSeries _BuildOutput(TimeSeries simulatedData, object userState = null)
        {
            int predictFuture = 0;


            double[] fstar;
            double[] Vfsar;
            double logpyX;

            double[] targets = GetDataArray(simulatedData);
            Predict(targets, predictFuture, mSigma0, mLambda, mSigmaN, out fstar, out Vfsar, out logpyX);

            TimeSeries ft = new TimeSeries();
            for (int i = 0; i < targets.Length; ++i)
            {
                if (mBuildOutputType == BuildOutputType.Mean)
                {
                    ft.Add(simulatedData.TimeStamp(i), fstar[i], true);
                }
                else
                {
                    ft.Add(simulatedData.TimeStamp(i), Vfsar[i], true);
                }
            }
            return ft;
        }

        /// <summary>
        /// GP Regression Method
        /// </summary>
        /// <param name="targets">the actual y value of the data points</param>
        /// <param name="future_point_count">the future point count to predict</param>
        /// <param name="sigma_0">amplitude of SE covariance</param>
        /// <param name="lambda">length scale of SE covariance</param>
        /// <param name="sigma_n">standard deviation of output noise</param>
        /// <param name="fstar">mean of GP</param>
        /// <param name="Vfstar">variance of GP</param>
        /// <param name="logpyX">log p(y|X)</param>
        public static void Predict(double[] targets, int future_point_count, double sigma_0, double lambda, double sigma_n, out double[] fstar, out double[] Vfstar, out double logpyX)
        {
            int n = targets.Length;
            double[,] cov = GetDataCovar(n, sigma_0, lambda, sigma_n);

            double mean = targets.Sum() / n;

            for (int i = 0; i < targets.Length; i++)
            {
                targets[i] -= mean;
            }

            //column matrix y
            SparseMatrix y = new SparseMatrix(n, 1, targets);

            SparseMatrix K = new SparseMatrix(cov);


            //identity matrix for I
            SparseMatrix I = SparseMatrix.Identity(targets.Length);

            //choleski(K + sigman^2*I)
            var temp_matrix = K.Add(I.Multiply(System.Math.Pow(sigma_n, 2)));
            var cholesky = temp_matrix.Cholesky();
            var L = cholesky.Factor;


            //inverse of L_transpose()
            //var L2 = L.LU().L;
            var L_transpose_inverse = L.Transpose().Inverse();
            //inverse of L
            var L_inverse = L.Inverse();

            //alpha = L'\(L\y)
            var alpha = L_transpose_inverse.Multiply(L_inverse).Multiply(y);

            double L_diag = 0.0;

            for (int i = 0; i < L.ColumnCount; i++)
            {
                L_diag += System.Math.Log(L[i, i]);
            }

            logpyX = -y.Transpose().Multiply(alpha).Multiply(0.5)[0, 0] - L_diag - future_point_count * System.Math.Log(2 * System.Math.PI) * 0.5;


            fstar = new double[targets.Length + future_point_count];
            Vfstar = new double[targets.Length + future_point_count];

            for (int i = 0; i < n + future_point_count; i++)
            {
                double[] kstar = new double[targets.Length];

                for (int j = 0; j < n; j++)
                {
                    kstar[j] = GetCov_SE(j, i, sigma_0, lambda, sigma_n);
                }

                var column_vector_kstar = new SparseMatrix(n, 1, kstar);

                //f*=k_*^T * alpha
                fstar[i] = column_vector_kstar.Transpose().Multiply(alpha)[0, 0];
                fstar[i] += mean;

                //v = L\k_*
                var v = L_inverse.Multiply(column_vector_kstar);

                //V[fstar] = k(x_*,x_*) - v^T*v
                Vfstar[i] = GetCov_SE(i, i, sigma_0, lambda, sigma_n) - v.Transpose().Multiply(v)[0, 0] + System.Math.Pow(sigma_n, 2);
            }
        }

        public override Predictor Clone()
        {
            GaussianDistributionPredictor p = new GaussianDistributionPredictor();
            p.WindowSize = mWindowSize;

            return p;
        }

        private static double GetCov_SE(int i, int j, double sigma_0, double lambda, double sigma_n)
        {
            double covar = System.Math.Pow(sigma_0, 2.0) * System.Math.Exp(-1.0 * (System.Math.Pow(i - j, 2.0) / (2 * System.Math.Pow(lambda, 2.0))));

            if (i == j)
            {
                covar += System.Math.Pow(sigma_n, 2);
            }

            return covar;
        }

        public double[] GetDataArray(TimeSeries ts)
        {
            int N = ts.Count;
            double[] data = new double[N];
            for (int i = 0; i < N; ++i)
            {
                data[i] = ts[i];
            }
            return data;
        }

        public static double[,] GetDataCovar(int N, double sigma_0, double lambda, double sigma_n)
        {
            double[,] cov = new double[N, N];
            for (int i = 0; i != N; ++i)
            {
                for (int j = 0; j != N; ++j)
                {
                    cov[i, j] = GetCov_SE(i, j, sigma_0, lambda, sigma_n);
                }
            }
            return cov;
        }





        public void predict(TimeSeries simulatedData, int predictFuture)
        {

        }
    }
}
