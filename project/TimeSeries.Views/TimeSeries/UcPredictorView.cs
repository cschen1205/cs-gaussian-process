using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace SimuKit.ML.Views.TimeSeries
{
    using ABMath.ModelFramework.Data;
    using ABMath.ModelFramework.Models;
    using SimuKit.ML.TimeSeries;

    public partial class UcPredictorView : UserControl
    {
        private ARMAPredictor mARMA = null;
        private MLPPredictor mMLP = null;
        private BasicNetworkPredictor mBasicNetwork = null;
        private SVMPredictor mSVM = null;
        private RBFNetworkPredictor mRBFNetwork = null;
        private GaussianDistributionPredictor mGaussianDistribution = null;
        private GeneticProgrammingPredictor mGeneticProgrammingPredictor = null;
        private GaussianProcessPredictor mGaussianProcess = null;

        public UcPredictorView()
        {
            InitializeComponent();

            if (DesignMode) return;

            ARMAPredictor p0 = ARMA;
            txtAROrder_ARMA.Text = string.Format("{0}", p0.AROrder);
            txtMAOrder_ARMA.Text = string.Format("{0}", p0.MAOrder);
            txtOptIter_ARMA.Text = string.Format("{0}", p0.OptIters);
            txtLDSIter_ARMA.Text = string.Format("{0}", p0.LDSIters);

            cboMLPTrainingMethod.DataSource = Enum.GetValues(typeof(MLPPredictor.TrainingMethod));
            MLPPredictor p1 = MLP;
            txtMLPNumberHiddenLayer.Text = string.Format("{0}", p1.NumberHiddenLayer);
            txtMLPWindowSize.Text = string.Format("{0}", p1.WindowSize);
            txtMLPMaxEpoch.Text = p1.MaxEpoch.ToString();
            txtMLPMaxError.Text = p1.MaxError.ToString();
            cboMLPTrainingMethod.SelectedItem = p1.Method;

            cboBasicNetworkTrainingMethod.DataSource = Enum.GetValues(typeof(BasicNetworkPredictor.TrainingMethod));
            BasicNetworkPredictor p2 = BasicNetwork;
            txtBasicNetworkNumberHiddenLayer.Text = p2.NumberHiddenLayer.ToString();
            txtBasicNetworkWindowSize.Text = string.Format("{0}", p2.WindowSize);
            txtBasicNetworkMaxEpoch.Text = p2.MaxEpoch.ToString();
            txtBasicNetworkMaxError.Text = p2.MaxError.ToString();
            txtBasicNetworkNormalizedHigh.Text = p2.NormalizedHigh.ToString();
            txtBasicNetworkNormalizedLow.Text = p2.NormalizedLow.ToString();
            cboBasicNetworkTrainingMethod.SelectedItem = p2.Method;

            SVMPredictor p3 = SVM;
            txtSVMWindowSize.Text = string.Format("{0}", p3.WindowSize);
            txtSVMNormalizedHigh.Text = p3.NormalizedHigh.ToString();
            txtSVMNormalizedLow.Text = p3.NormalizedLow.ToString();

            RBFNetworkPredictor p4 = RBFNetwork;
            txtRBFNetworkWindowSize.Text = string.Format("{0}", p4.WindowSize);
            txtRBFNetworkNormalizedHigh.Text = p4.NormalizedHigh.ToString();
            txtRBFNetworkNormalizedLow.Text = p4.NormalizedLow.ToString();

            GaussianDistributionPredictor p5 = GaussianDistribution;
            chkGaussinDistributionPositiveValueOnly.Checked = p5.PositveValueOnly;

            GeneticProgrammingPredictor p6 = GeneticProgramming;
            txtGeneticProgrammingPopSize.Text = p6.PopSize.ToString();
            txtGeneticProgrammingWindowSize.Text = p6.WindowSize.ToString();
            txtGeneticProgrammingMaxEpoch.Text = p6.MaxEpoch.ToString();

            GaussianProcessPredictor p7 = GaussianProcess;
            txtGaussianProcessLambda.Text = p7.Lambda.ToString();
            txtGaussianProcessSigma0.Text = p7.Sigma0.ToString();
            txtGaussianProcessSigmaN.Text = p7.SigmaN.ToString();
        }



        public Predictor GetPredictor(int selected_index)
        {
            if (selected_index == 0)
            {
                ARMAPredictor p = ARMA;

                int result = 0;
                if (int.TryParse(txtAROrder_ARMA.Text, out result))
                {
                    p.AROrder = result;
                }
                if (int.TryParse(txtMAOrder_ARMA.Text, out result))
                {
                    p.MAOrder = result;
                }
                if (int.TryParse(txtLDSIter_ARMA.Text, out result))
                {
                    p.LDSIters = result;
                }
                if (int.TryParse(txtOptIter_ARMA.Text, out result))
                {
                    p.OptIters = result;
                }

                return mARMA;
            }
            else if (selected_index == 1)
            {
                MLPPredictor p = MLP;

                int result = 0;
                if (int.TryParse(txtMLPNumberHiddenLayer.Text, out result))
                {
                    p.NumberHiddenLayer = result;
                }
                if (int.TryParse(txtMLPWindowSize.Text, out result))
                {
                    p.WindowSize = result;
                }
                if (int.TryParse(txtMLPMaxEpoch.Text, out result))
                {
                    p.MaxEpoch = result;
                }

                double dresult = 0;
                if (double.TryParse(txtMLPMaxError.Text, out dresult))
                {
                    p.MaxError = dresult;
                }

                p.Method = (MLPPredictor.TrainingMethod)cboMLPTrainingMethod.SelectedItem;


                return mMLP;
            }
            else if (selected_index == 2)
            {
                BasicNetworkPredictor p = BasicNetwork;

                int result = 0;
                if (int.TryParse(txtBasicNetworkNumberHiddenLayer.Text, out result))
                {
                    p.NumberHiddenLayer = result;
                }
                if (int.TryParse(txtBasicNetworkWindowSize.Text, out result))
                {
                    p.WindowSize = result;
                }
                if (int.TryParse(txtBasicNetworkMaxEpoch.Text, out result))
                {
                    p.MaxEpoch = result;
                }

                double dresult = 0;
                if (double.TryParse(txtBasicNetworkMaxError.Text, out dresult))
                {
                    p.MaxError = dresult;
                }
                if (double.TryParse(txtBasicNetworkNormalizedHigh.Text, out dresult))
                {
                    p.NormalizedHigh = dresult;
                }
                if (double.TryParse(txtBasicNetworkNormalizedLow.Text, out dresult))
                {
                    p.NormalizedLow = dresult;
                }

                p.Method = (BasicNetworkPredictor.TrainingMethod)cboBasicNetworkTrainingMethod.SelectedItem;
                return mBasicNetwork;
            }
            else if (selected_index == 3)
            {
                SVMPredictor p = SVM;


                int result = 0;

                if (int.TryParse(txtSVMWindowSize.Text, out result))
                {
                    p.WindowSize = result;
                }

                double dresult = 0;
                if (double.TryParse(txtSVMNormalizedHigh.Text, out dresult))
                {
                    p.NormalizedHigh = dresult;
                }
                if (double.TryParse(txtSVMNormalizedLow.Text, out dresult))
                {
                    p.NormalizedLow = dresult;
                }
                return mSVM;
            }
            else if (selected_index == 4)
            {
                RBFNetworkPredictor p = RBFNetwork;

                int result = 0;

                if (int.TryParse(txtRBFNetworkWindowSize.Text, out result))
                {
                    p.WindowSize = result;
                }

                double dresult = 0;
                if (double.TryParse(txtRBFNetworkNormalizedHigh.Text, out dresult))
                {
                    p.NormalizedHigh = dresult;
                }
                if (double.TryParse(txtRBFNetworkNormalizedLow.Text, out dresult))
                {
                    p.NormalizedLow = dresult;
                }
                return mRBFNetwork;
            }
            else if (selected_index == 5)
            {
                GaussianDistributionPredictor p = GaussianDistribution;
                p.PositveValueOnly = chkGaussinDistributionPositiveValueOnly.Checked;
                return mGaussianDistribution;
            }
            else if (selected_index == 6)
            {
                GeneticProgrammingPredictor p = GeneticProgramming;

                int result = 0;

                if (int.TryParse(txtGeneticProgrammingWindowSize.Text, out result))
                {
                    p.WindowSize = result;
                }

                if (int.TryParse(txtGeneticProgrammingPopSize.Text, out result))
                {
                    p.PopSize = result;
                }


                if (int.TryParse(txtGeneticProgrammingMaxEpoch.Text, out result))
                {
                    p.MaxEpoch = result;
                }

                return mGeneticProgrammingPredictor;
            }
            else if (selected_index == 7)
            {
                GaussianProcessPredictor p = GaussianProcess;

                double result = 0;

                if (double.TryParse(txtGaussianProcessSigma0.Text, out result))
                {
                    p.Sigma0 = result;
                }

                if (double.TryParse(txtGaussianProcessLambda.Text, out result))
                {
                    p.Lambda = result;
                }

                if (double.TryParse(txtGaussianProcessSigmaN.Text, out result))
                {
                    p.SigmaN = result;
                }

                return mGaussianProcess;
            }
            else
            {
                return null;
            }
        }

        public void Switch2ARMA()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[0];
        }

        public void Switch2MLP()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[1];
        }

        public void Switch2BasicNetwork()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[2];
        }

        public void Switch2SVM()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[3];
        }

        public void Switch2RBFNetwork()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[4];
        }

        public void Switch2GaussianDistribution()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[5];
        }

        public void Switch2GeneticProgramming()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[6];
        }

        public void Switch2GaussianProcess()
        {
            tcPredictors.SelectedTab = tcPredictors.TabPages[7];
        }

        public Predictor SelectedPredictor
        {
            get
            {
                Predictor p = GetPredictor(tcPredictors.SelectedIndex);
                return p;
            }
            set
            {
                if (value == ARMA)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[0];
                }
                else if (value == MLP)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[1];
                }
                else if (value == BasicNetwork)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[2];
                }
                else if (value == SVM)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[3];
                }
                else if (value == RBFNetwork)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[4];
                }
                else if (value == GaussianDistribution)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[5];
                }
                else if (value == GeneticProgramming)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[6];
                }
                else if (value == GaussianProcess)
                {
                    tcPredictors.SelectedTab = tcPredictors.TabPages[7];
                }
            }
        }

        private void OnOutputBuilt(Predictor p)
        {
            UpdatePredictorStat(p);
        }


        public ARMAPredictor ARMA
        {
            get
            {
                if (mARMA == null)
                {
                    mARMA = new ARMAPredictor();
                    mARMA.OutputBuilt += OnOutputBuilt;
                }
                return mARMA;
            }
        }

        public GaussianProcessPredictor GaussianProcess
        {
            get
            {
                if (mGaussianProcess == null)
                {
                    mGaussianProcess = new GaussianProcessPredictor();
                    mGaussianProcess.OutputBuilt += OnOutputBuilt;
                }
                return mGaussianProcess;
            }
        }

        public GeneticProgrammingPredictor GeneticProgramming
        {
            get
            {
                if (mGeneticProgrammingPredictor == null)
                {
                    mGeneticProgrammingPredictor = new GeneticProgrammingPredictor();
                    mGeneticProgrammingPredictor.OutputBuilt += OnOutputBuilt;
                }
                return mGeneticProgrammingPredictor;
            }
        }

        public GaussianDistributionPredictor GaussianDistribution
        {
            get
            {
                if (mGaussianDistribution == null)
                {
                    mGaussianDistribution = new GaussianDistributionPredictor();
                    mGaussianDistribution.OutputBuilt += OnOutputBuilt;
                }
                return mGaussianDistribution;
            }
        }


        public MLPPredictor MLP
        {
            get
            {
                if (mMLP == null)
                {
                    mMLP = new MLPPredictor();
                    mMLP.OutputBuilt += OnOutputBuilt;
                }
                return mMLP;
            }
        }

        public BasicNetworkPredictor BasicNetwork
        {
            get
            {
                if (mBasicNetwork == null)
                {
                    mBasicNetwork = new BasicNetworkPredictor();
                    mBasicNetwork.OutputBuilt += OnOutputBuilt;
                }
                return mBasicNetwork;
            }
        }

        public RBFNetworkPredictor RBFNetwork
        {
            get
            {
                if (mRBFNetwork == null)
                {
                    mRBFNetwork = new RBFNetworkPredictor();
                    mRBFNetwork.OutputBuilt += OnOutputBuilt;
                }
                return mRBFNetwork;
            }
        }

        public SVMPredictor SVM
        {
            get
            {
                if (mSVM == null)
                {
                    mSVM = new SVMPredictor();
                    mSVM.OutputBuilt += OnOutputBuilt;
                }
                return mSVM;
            }
        }

        private void UcPredictorView_Load(object sender, EventArgs e)
        {

        }

        private void tcPredictors_SelectedIndexChanged(object sender, EventArgs e)
        {
            Predictor p = SelectedPredictor;
            UpdatePredictorStat(p);
        }

        private void UpdatePredictorStat(Predictor p)
        {
            PredictorStat stat = p.Stat;
            if (stat.Computed)
            {
                lblMAE.Text = string.Format("MAE: {0}", stat.MAE.ToString("0.00"));
                lblMAPE.Text = string.Format("MAPE: {0}", stat.MAPE.ToString("0.00"));
                lblMASE.Text = string.Format("MASE: {0}", stat.MASE.ToString("0.00"));
                lblMSE.Text = string.Format("MSE: {0}", stat.MSE.ToString("0.00"));
                lblSMAPE.Text = string.Format("SMAPE: {0}", stat.SMAPE.ToString("0.00"));
                lblRMSE.Text = string.Format("RMSE: {0}", stat.RMSE.ToString("0.00"));
            }
            else
            {
                lblMAE.Text = "MAE";
                lblMAPE.Text = "MAPE";
                lblMASE.Text = "MASE";
                lblMSE.Text = "MSE";
                lblSMAPE.Text = "SMAPE";
                lblRMSE.Text = "RMSE";
            }
        }

        private void lblMSE_Click(object sender, EventArgs e)
        {
            MessageBox.Show(PredictorStat.DescMSE, "MSE");
        }

        private void lblRMSE_Click(object sender, EventArgs e)
        {
            MessageBox.Show(PredictorStat.DescRMSE, "RMSE");
        }

        private void lblMAE_Click(object sender, EventArgs e)
        {
            MessageBox.Show(PredictorStat.DescMAE, "MAE");
        }

        private void lblMASE_Click(object sender, EventArgs e)
        {
            MessageBox.Show(PredictorStat.DescMASE, "MASE");
        }

        private void lblMAPE_Click(object sender, EventArgs e)
        {
            MessageBox.Show(PredictorStat.DescMAPE, "MAPE");
        }

        private void lblSMAPE_Click(object sender, EventArgs e)
        {
            MessageBox.Show(PredictorStat.DescSMAPE, "SMAPE");
        }
    }
}
