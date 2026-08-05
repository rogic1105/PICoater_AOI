using System.Drawing;
using System.Linq;
using System.Threading;
using System.Windows.Forms.DataVisualization.Charting;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    public class ColumnCurveChartHelperTests
    {
        [Test]
        public void UpdateDataAndView_SameDisplayLength_ReusesDataPointsAndUpdatesValues()
        {
            using (var chart = new Chart())
            {
                var helper = new ColumnCurveChartHelper(chart);
                helper.SetOps(1000);
                helper.UpdateDataAndView(
                    new float[] { 10, 20, 30 },
                    new float[] { 40, 50, 60 },
                    5,
                    double.NaN,
                    double.NaN);

                DataPoint firstMeanPoint = chart.Series["Mean"].Points[0];
                DataPoint firstMaxPoint = chart.Series["Max"].Points[0];

                helper.UpdateDataAndView(
                    new float[] { 70, 80, 90 },
                    new float[] { 100, 110, 120 },
                    7,
                    double.NaN,
                    double.NaN);

                Assert.That(chart.Series["Mean"].Points[0], Is.SameAs(firstMeanPoint));
                Assert.That(chart.Series["Max"].Points[0], Is.SameAs(firstMaxPoint));
                Assert.That(firstMeanPoint.XValue, Is.EqualTo(7));
                Assert.That(firstMeanPoint.YValues[0], Is.EqualTo(70.0 / 255.0).Within(1e-9));
                Assert.That(firstMaxPoint.YValues[0], Is.EqualTo(100.0 / 255.0).Within(1e-9));
            }
        }

        [Test]
        public void UpdateData_WideInput_UsesPixelBudgetAndPreservesMaxPeak()
        {
            using (var chart = new Chart { Size = new Size(800, 100) })
            {
                var helper = new ColumnCurveChartHelper(chart);
                var mean = new float[16384];
                var max = new float[16384];
                max[8123] = 255;

                helper.UpdateData(mean, max, 0);

                Assert.That(helper.DisplayPointCount, Is.LessThanOrEqualTo(400));
                Assert.That(chart.Series["Mean"].Points.Count, Is.EqualTo(helper.DisplayPointCount));
                Assert.That(chart.Series["Max"].Points.Max(p => p.YValues[0]), Is.EqualTo(1.0));
            }
        }

        [Test]
        public void SingleMeanPeak_BelowThreshold_RemainsVisibleAndPassesVerdict()
        {
            using (var chart = new Chart { Size = new Size(200, 100) })
            {
                var helper = new ColumnCurveChartHelper(chart);
                var mean = new float[1024];
                mean[511] = 178.5f; // 0.70 after /255; one narrow point in a display bucket.

                helper.UpdateData(mean, null, 0);
                var threshold = new ThresholdContext(
                    1f, 0.80f, 2f,
                    1f, 0.20f, 0.60f,
                    ColumnCurveDisplayMode.Mean);
                ColumnVerdictEvaluation verdict = threshold.EvaluateColumn(
                    (float)helper.DisplayMeanPeak, float.NaN, 1f);

                Assert.That(helper.DisplayPointCount, Is.LessThan(mean.Length));
                Assert.That(helper.DisplayMeanPeak, Is.EqualTo(0.70).Within(0.0001));
                Assert.That(verdict.IsFail, Is.False);
            }
        }

        [Test]
        public void UpdateDataAndView_DataExtentChanges_KeepsAxisOwnedByViewport()
        {
            using (var chart = new Chart { Size = new Size(800, 120) })
            {
                var helper = new ColumnCurveChartHelper(chart);
                helper.SetOps(1000);

                helper.UpdateDataAndView(
                    new float[100], new float[100], 0, 20, 40);
                var axis = chart.ChartAreas[0].AxisX;
                double minimum = axis.Minimum;
                double maximum = axis.Maximum;

                helper.UpdateDataAndView(
                    new float[1000], new float[1000], 0, 20, 40);

                Assert.That(axis.Minimum, Is.EqualTo(minimum).Within(1e-9));
                Assert.That(axis.Maximum, Is.EqualTo(maximum).Within(1e-9));
                Assert.That(axis.Minimum, Is.EqualTo(axis.ScaleView.ViewMinimum).Within(1e-9));
                Assert.That(axis.Maximum, Is.EqualTo(axis.ScaleView.ViewMaximum).Within(1e-9));
            }
        }
    }
}
