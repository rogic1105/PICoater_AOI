using System.Threading;
using System.Windows.Forms.DataVisualization.Charting;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.UI.Managers;
using NUnit.Framework;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    public class RowCurveSyncCoordinatorTests
    {
        [Test]
        public void SuspendUntilNextData_NewDataWaitsForNewViewRange()
        {
            using (var chart = new Chart())
            {
                var helper = new RowCurveChartHelper(chart);
                var display = new RowCurveDisplayAdapter(
                    helper, () => VerticalDisplayDirection.BottomToTop);
                var sync = new RowCurveSyncCoordinator(display);
                sync.SetRowPitch(1);

                sync.SetViewRange(0, 10);
                sync.UpdateData(new float[10], new float[10], requireViewRange: true);
                Assert.That(helper.TotalMm, Is.EqualTo(10));

                sync.SuspendUntilNextData();
                sync.UpdateData(new float[20], new float[20], requireViewRange: true);
                sync.Resume();
                Assert.That(helper.TotalMm, Is.EqualTo(10),
                    "new data must remain pending instead of using the previous image range");

                sync.SetViewRange(0, 20);
                Assert.That(helper.TotalMm, Is.EqualTo(20));
            }
        }

        [Test]
        public void UpdateData_ReportsAppliedOnlyAfterPendingViewRangeIsPublished()
        {
            using (var chart = new Chart())
            {
                var helper = new RowCurveChartHelper(chart);
                var display = new RowCurveDisplayAdapter(
                    helper, () => VerticalDisplayDirection.BottomToTop);
                var sync = new RowCurveSyncCoordinator(display);
                int applied = 0;
                sync.DataAccepted += () => applied++;

                sync.UpdateData(new float[20], new float[20], requireViewRange: true);
                Assert.That(applied, Is.Zero);

                sync.SetViewRange(0, 20);
                Assert.That(applied, Is.EqualTo(1));
            }
        }

        [Test]
        public void UpdateDataAndView_DataLengthChanges_KeepsAxisOwnedByViewport()
        {
            using (var chart = new Chart())
            {
                var helper = new RowCurveChartHelper(chart)
                {
                    ZeroAtTop = false
                };
                helper.SetRowPitch(1);
                helper.UpdateDataAndViewRange(
                    new float[100], new float[100], 40, 20);
                var axis = chart.ChartAreas[0].AxisY;
                double minimum = axis.Minimum;
                double maximum = axis.Maximum;

                helper.UpdateDataAndViewRange(
                    new float[1000], new float[1000], 40, 20);

                Assert.That(axis.Minimum, Is.EqualTo(minimum).Within(1e-9));
                Assert.That(axis.Maximum, Is.EqualTo(maximum).Within(1e-9));
                Assert.That(axis.Minimum, Is.EqualTo(axis.ScaleView.ViewMinimum).Within(1e-9));
                Assert.That(axis.Maximum, Is.EqualTo(axis.ScaleView.ViewMaximum).Within(1e-9));
            }
        }

        [Test]
        public void UpdateDataPreservingView_ChangesValuesWithoutMovingPhysicalRange()
        {
            using (var chart = new Chart())
            {
                var helper = new RowCurveChartHelper(chart)
                {
                    ZeroAtTop = false
                };
                helper.SetRowPitch(1);
                helper.UpdateDataAndViewRange(
                    new float[100], new float[100], 40, 20);
                var axis = chart.ChartAreas[0].AxisY;
                double minimum = axis.Minimum;
                double maximum = axis.Maximum;
                double viewMinimum = axis.ScaleView.ViewMinimum;
                double viewMaximum = axis.ScaleView.ViewMaximum;

                helper.UpdateDataPreservingView(
                    new float[1000], new float[1000]);

                Assert.That(axis.Minimum, Is.EqualTo(minimum).Within(1e-9));
                Assert.That(axis.Maximum, Is.EqualTo(maximum).Within(1e-9));
                Assert.That(axis.ScaleView.ViewMinimum, Is.EqualTo(viewMinimum).Within(1e-9));
                Assert.That(axis.ScaleView.ViewMaximum, Is.EqualTo(viewMaximum).Within(1e-9));
                Assert.That(helper.TotalMm, Is.EqualTo(1000));
            }
        }
    }
}
