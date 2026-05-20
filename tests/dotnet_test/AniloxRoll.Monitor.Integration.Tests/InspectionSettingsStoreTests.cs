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
        public void SaveAndLoad_PersistsPlcSettings()
        {
            var settings = new InspectionSettings
            {
                PlcEnabled = true,
                PlcIp = "10.20.30.40",
                PlcPort = 1502
            };

            InspectionSettingsStore.Save(settings);
            var loaded = InspectionSettingsStore.Load();

            Assert.That(loaded.PlcEnabled, Is.True);
            Assert.That(loaded.PlcIp, Is.EqualTo("10.20.30.40"));
            Assert.That(loaded.PlcPort, Is.EqualTo(1502));
        }
    }
}
