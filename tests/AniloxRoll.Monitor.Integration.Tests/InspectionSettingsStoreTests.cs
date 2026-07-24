using System;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Data;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionSettingsStoreTests
    {
        private string _configPath;
        private string _backupPath;

        [SetUp]
        public void SetUp()
        {
            SettingsStoreHelper.DrainIssues();
            _configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\inspection-settings.json");
            _backupPath = _configPath + ".bak_test";

            Directory.CreateDirectory(Path.GetDirectoryName(_configPath));

            if (File.Exists(_backupPath))
                File.Delete(_backupPath);

            if (File.Exists(_configPath))
                File.Move(_configPath, _backupPath);
        }

        [TearDown]
        public void TearDown()
        {
            if (File.Exists(_configPath))
                File.Delete(_configPath);

            if (File.Exists(_backupPath))
                File.Move(_backupPath, _configPath);
        }

        [Test]
        public void SaveAndLoad_PersistsIoSettings()
        {
            var settings = new InspectionSettings
            {
                IoEnabled = true,
                IoIp = "10.20.30.40",
                IoPort = 1502
            };

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.IoEnabled, Is.True);
            Assert.That(loaded.IoIp, Is.EqualTo("10.20.30.40"));
            Assert.That(loaded.IoPort, Is.EqualTo(1502));
        }

        [Test]
        public void SaveAndLoad_PersistsTimeSettings()
        {
            var settings = new InspectionSettings();
            settings.BackgroundSampleSeconds = 4;
            settings.GrabLimitSeconds = 17;

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.BackgroundSampleSeconds, Is.EqualTo(4));
            Assert.That(loaded.GrabLimitSeconds, Is.EqualTo(17));
        }

        [Test]
        public void SaveAndLoad_PersistsInspectionAndWaterfallSettings()
        {
            var settings = new InspectionSettings();
            settings.RidgeSigma = 12.5f;
            settings.ImageView.WaterfallTotalHeight = 42000;
            settings.ImageView.WaterfallFullMode = WaterfallFullMode.Ring;

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.RidgeSigma, Is.EqualTo(12.5f).Within(0.001f));
            Assert.That(loaded.ImageView.WaterfallTotalHeight, Is.EqualTo(42000));
            Assert.That(loaded.ImageView.WaterfallFullMode, Is.EqualTo(WaterfallFullMode.Ring));
        }

        [Test]
        public void Load_WhenConfigIsMissing_DefaultsMainDisplayToWaterfall()
        {
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.he_MainDisplay, Is.EqualTo(MainDisplayMode.Waterfall));
        }

        [Test]
        public void Load_WhenConfigIsMissing_UsesPackedCaptureRoot()
        {
            var loaded = InspectionSettingsStore.Load();

            Assert.That(
                loaded.CaptureRootPath,
                Is.EqualTo(@"D:\Anilox\Captures_pack").IgnoreCase);
            Assert.That(
                loaded.RemotePath,
                Is.EqualTo(@"\\192.168.10.20\Anilox\Captures_pack").IgnoreCase);
        }

        [Test]
        public void Load_LegacyDefaultRemotePath_UpgradesToPackedRoot()
        {
            File.WriteAllText(
                _configPath,
                "{\"Storage\":{\"AniloxRootPath\":\"D:\\\\Anilox\"," +
                "\"RemotePath\":\"\\\\\\\\192.168.10.20\\\\Anilox\\\\Captures\"}}");

            var loaded = InspectionSettingsStore.Load();

            Assert.That(
                loaded.RemotePath,
                Is.EqualTo(@"\\192.168.10.20\Anilox\Captures_pack").IgnoreCase);
        }

        [Test]
        public void SaveAndLoad_PersistsLogRetentionHours()
        {
            var settings = new InspectionSettings();
            settings.LogRetentionHours = 72;

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.LogRetentionHours, Is.EqualTo(72));
        }

        [Test]
        public void SaveAndLoad_PersistsLogRecordingMode()
        {
            var settings = new InspectionSettings
            {
                LogMode = LogRecordingMode.FlowVerification
            };

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.LogMode, Is.EqualTo(LogRecordingMode.FlowVerification));
        }

        [Test]
        public void SaveAndLoad_PersistsEnhanceHeatmap()
        {
            var settings = new InspectionSettings
            {
                EnhanceHeatmap = EnhanceHeatmapMode.BlueYellowRed
            };

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.EnhanceHeatmap, Is.EqualTo(EnhanceHeatmapMode.BlueYellowRed));
        }

        [Test]
        public void Load_LegacyHeatmapBooleanTrue_MapsToCold()
        {
            File.WriteAllText(_configPath,
                "{\"ImageView\":{\"EnableEnhanceHeatmap\":true}}");

            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.EnhanceHeatmap, Is.EqualTo(EnhanceHeatmapMode.Cold));
        }

        [Test]
        public void Load_LegacyBlueRedMode_MapsToBlueYellowRed()
        {
            File.WriteAllText(_configPath,
                "{\"ImageView\":{\"EnhanceHeatmap\":\"BlueRed\"}}");

            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.EnhanceHeatmap, Is.EqualTo(EnhanceHeatmapMode.BlueYellowRed));
        }

        [Test]
        public void Load_LegacyDebugUiActionLog_MapsToFullDiagnostic()
        {
            File.WriteAllText(_configPath,
                "{\"Storage\":{\"LogRetentionHours\":72},\"DebugUiActionLog\":true}");

            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.LogMode, Is.EqualTo(LogRecordingMode.FullDiagnostic));
            Assert.That(loaded.LogRetentionHours, Is.EqualTo(72));
        }

        [Test]
        public void Load_WhenConfigIsCorrupt_RebuildsDefaultsAndRecordsIssue()
        {
            File.WriteAllText(_configPath, "{ invalid json");

            var loaded = InspectionSettingsStore.Load();
            SettingsStoreIssue[] issues = SettingsStoreHelper.DrainIssues();

            Assert.That(loaded.he_MainDisplay, Is.EqualTo(MainDisplayMode.Waterfall));
            Assert.That(File.Exists(_configPath), Is.True);
            Assert.That(
                Array.Exists(issues, x => x.Kind == SettingsStoreIssueKind.RebuiltDefaults),
                Is.True);
        }

        [Test]
        public void SaveFailure_RaisesRuntimeIssueImmediately()
        {
            string parentFile = Path.Combine(
                Path.GetDirectoryName(_configPath), "not-a-directory");
            File.WriteAllText(parentFile, "block directory creation");
            SettingsStoreIssue observed = null;
            Action<SettingsStoreIssue> handler = issue => observed = issue;
            SettingsStoreHelper.IssueRaised += handler;
            try
            {
                SettingsStoreHelper.SaveJsonFile(
                    Path.Combine(parentFile, "settings.json"),
                    "{}",
                    "RuntimeIssueTest");
            }
            finally
            {
                SettingsStoreHelper.IssueRaised -= handler;
                File.Delete(parentFile);
            }

            Assert.That(observed, Is.Not.Null);
            Assert.That(observed.Kind, Is.EqualTo(SettingsStoreIssueKind.SaveFailed));
        }

        [Test]
        public void JsonConfigLoader_CorruptJson_ReturnsFallbackAndRecordsIssue()
        {
            string fileName = "json-loader-" + Guid.NewGuid().ToString("N") + ".json";
            string relativePath = Path.Combine("Config", fileName);
            string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, relativePath);
            File.WriteAllText(fullPath, "{ invalid json");
            var fallback = new AppModeConfig { StorageMinFreeGB = 321 };

            var loaded = JsonConfigLoader.LoadOrDefault(relativePath, fallback);
            SettingsStoreIssue[] issues = SettingsStoreHelper.DrainIssues();

            Assert.That(loaded, Is.SameAs(fallback));
            Assert.That(File.Exists(fullPath), Is.False);
            Assert.That(
                Array.Exists(
                    issues,
                    x => x.Kind == SettingsStoreIssueKind.RebuiltDefaults &&
                         string.Equals(x.Path, fullPath, StringComparison.OrdinalIgnoreCase)),
                Is.True);
        }
    }
}
