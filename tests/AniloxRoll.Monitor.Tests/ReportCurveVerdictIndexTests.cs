using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReportCurveVerdictIndexTests
    {
        [Test]
        public void ReplaceDetails_ChangesRootAndSettingsAsOneState()
        {
            var index = new ReportCurveVerdictIndex();
            ThresholdContext first = Context(0.5f);
            ThresholdContext second = Context(0.8f);

            index.ReplaceDetails("D:\\one", Details("g1"), first);

            Assert.That(index.IsCurrent("d:\\ONE"), Is.True);
            Assert.That(index.IsVerdictCurrent(first), Is.True);
            Assert.That(index.IsVerdictCurrent(second), Is.False);
            Assert.That(index.Details.ContainsKey("g1"), Is.True);

            index.ReplaceDetails("D:\\two", Details("g2"), second);

            Assert.That(index.IsCurrent("D:\\one"), Is.False);
            Assert.That(index.IsCurrent("D:\\two"), Is.True);
            Assert.That(index.IsVerdictCurrent(second), Is.True);
            Assert.That(index.Details.ContainsKey("g1"), Is.False);
            Assert.That(index.Details.ContainsKey("g2"), Is.True);
        }

        [Test]
        public void ReplaceDetails_WhenRootChanges_ClearsAllPeaks()
        {
            var index = new ReportCurveVerdictIndex();
            index.ReplaceDetails("D:\\one", Details("same-id"), Context(0.5f));
            index.ColumnPeaks["same-id"] = new ColumnCurvePeakRecord[1];
            index.RowPeaks["same-id"] = new RowCurvePeakRecord();

            index.ReplaceDetails("D:\\two", Details("same-id"), Context(0.5f));

            Assert.That(index.ColumnPeaks, Is.Empty);
            Assert.That(index.RowPeaks, Is.Empty);
            Assert.That(index.HasBothPeaks("same-id"), Is.False);
        }

        [Test]
        public void ReplaceDetails_WhenSameRoot_PrunesRemovedIdsAndKeepsSurvivors()
        {
            var index = new ReportCurveVerdictIndex();
            index.ReplaceDetails("D:\\one", Details("keep", "remove"), Context(0.5f));
            index.ColumnPeaks["keep"] = new ColumnCurvePeakRecord[1];
            index.RowPeaks["keep"] = new RowCurvePeakRecord();
            index.ColumnPeaks["remove"] = new ColumnCurvePeakRecord[1];
            index.RowPeaks["remove"] = new RowCurvePeakRecord();

            index.ReplaceDetails("d:\\ONE", Details("keep"), Context(0.8f));

            Assert.That(index.HasBothPeaks("keep"), Is.True);
            Assert.That(index.ColumnPeaks.ContainsKey("remove"), Is.False);
            Assert.That(index.RowPeaks.ContainsKey("remove"), Is.False);
        }

        [Test]
        public void Reset_ClearsDetailsPeaksAndIdentity()
        {
            var index = new ReportCurveVerdictIndex();
            index.ReplaceDetails("D:\\one", Details("g1"), Context(0.5f));
            index.ColumnPeaks["g1"] = new ColumnCurvePeakRecord[1];
            index.RowPeaks["g1"] = new RowCurvePeakRecord();

            index.Reset();

            Assert.That(index.IsCurrent("D:\\one"), Is.False);
            Assert.That(index.Details, Is.Empty);
            Assert.That(index.ColumnPeaks, Is.Empty);
            Assert.That(index.RowPeaks, Is.Empty);
        }

        [Test]
        public void Project_MarksAppliedSettingsAsCurrent()
        {
            var index = new ReportCurveVerdictIndex();
            ThresholdContext first = Context(0.5f);
            ThresholdContext second = Context(0.8f);
            index.ReplaceDetails("D:\\one", Details("g1"), first);

            Assert.That(index.IsVerdictCurrent(second), Is.False);

            index.Project(second);

            Assert.That(index.IsVerdictCurrent(second), Is.True);
            Assert.That(index.IsVerdictCurrent(first), Is.False);
        }

        private static IEnumerable<KeyValuePair<string, GrabDetail>> Details(params string[] grabIds)
        {
            var details = new Dictionary<string, GrabDetail>();
            foreach (string grabId in grabIds)
            {
                details[grabId] = new GrabDetail { GrabId = grabId };
            }
            return details;
        }

        private static ThresholdContext Context(float threshold)
        {
            return new ThresholdContext(
                1f, threshold, threshold,
                1f, threshold, threshold,
                ColumnCurveDisplayMode.Both, RidgeDirection.Both);
        }
    }
}
