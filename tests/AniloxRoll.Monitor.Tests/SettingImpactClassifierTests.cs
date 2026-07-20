using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class SettingImpactClassifierTests
    {
        [Test]
        public void KnownSettings_CoverEveryEditablePropertyGridSetting()
        {
            var editableNames = TypeDescriptor.GetProperties(typeof(InspectionSettings))
                .Cast<PropertyDescriptor>()
                .Where(property => property.IsBrowsable && !property.IsReadOnly)
                .Select(property => property.Name)
                .OrderBy(name => name)
                .ToArray();
            string[] knownNames = SettingImpactClassifier.KnownSettingNames
                .OrderBy(name => name)
                .ToArray();

            CollectionAssert.AreEqual(editableNames, knownNames);
        }

        private static IEnumerable<TestCaseData> ExactRoutes()
        {
            yield return Route(nameof(InspectionSettings.IoIp), SettingFeatureOwner.Io);
            yield return Route(nameof(InspectionSettings.LightBrightness), SettingFeatureOwner.Light);
            yield return Route(nameof(InspectionSettings.LocalMinFreeGB), SettingFeatureOwner.Storage);
            yield return Route(nameof(InspectionSettings.he_MainDisplay), SettingFeatureOwner.LiveLayout);
            yield return Route(nameof(InspectionSettings.gb_ChartScaleMode), SettingFeatureOwner.ChartScale);
            yield return Route(nameof(InspectionSettings.fb_BackgroundSampleSeconds), SettingFeatureOwner.None);
            yield return Route(
                nameof(InspectionSettings.ai_OpsSpeed),
                SettingFeatureOwner.None,
                SettingImpact.RowPitch);
            yield return Route(
                nameof(InspectionSettings.de_RidgeSigma),
                SettingFeatureOwner.None,
                SettingImpact.CapturePolicy);
            yield return Route(
                nameof(InspectionSettings.ec_ErrorValueMeanV),
                SettingFeatureOwner.DataStats,
                SettingImpact.InspectionService |
                SettingImpact.ColumnThresholds |
                SettingImpact.ReviewCurves);
            yield return Route(
                nameof(InspectionSettings.ee_ErrorValueMeanH),
                SettingFeatureOwner.DataStats,
                SettingImpact.RowThresholds |
                SettingImpact.ReviewCurves);
        }

        [TestCaseSource(nameof(ExactRoutes))]
        public void Classify_ReturnsExactOwnerAndImpacts(
            string name,
            string expectedOwner,
            int expectedImpacts)
        {
            SettingRoute route = SettingImpactClassifier.Classify(name);

            Assert.That(route.Owner.ToString(), Is.EqualTo(expectedOwner));
            Assert.That((int)route.Impacts, Is.EqualTo(expectedImpacts));
        }

        [Test]
        public void Classify_UnknownSetting_HasNoRuntimeSideEffects()
        {
            SettingRoute route = SettingImpactClassifier.Classify("FutureSetting");

            Assert.That(route.Owner, Is.EqualTo(SettingFeatureOwner.None));
            Assert.That(route.Impacts, Is.EqualTo(SettingImpact.None));
        }

        [Test]
        public void ToLogText_UsesStableMachineReadableFormat()
        {
            var route = new SettingRoute(
                SettingFeatureOwner.DataStats,
                SettingImpact.InspectionService | SettingImpact.ColumnThresholds);

            Assert.That(
                route.ToLogText(),
                Is.EqualTo("owner=DataStats effects=InspectionService+ColumnThresholds"));
        }

        private static TestCaseData Route(
            string name,
            SettingFeatureOwner owner,
            SettingImpact impacts = SettingImpact.None)
        {
            return new TestCaseData(name, owner.ToString(), (int)impacts).SetName($"Route_{name}");
        }
    }
}
