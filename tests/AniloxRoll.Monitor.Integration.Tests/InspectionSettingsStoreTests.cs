using System;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Data;

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
    }
}
