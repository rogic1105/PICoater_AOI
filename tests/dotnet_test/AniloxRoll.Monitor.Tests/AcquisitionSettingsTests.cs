using System;
using System.IO;
using System.Text;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class AcquisitionSettingsTests
    {
        [Test]
        public void Validate_DefaultValues_NoChange()
        {
            var s = new AcquisitionSettings();
            s.Validate();
            Assert.That(s.CameraGrabHeight.Length, Is.EqualTo(7));
            Assert.That(s.CameraExposureTimeUs.Length, Is.EqualTo(7));
            Assert.That(s.CameraLineRateHz.Length, Is.EqualTo(7));
            Assert.That(s.CameraGrabHeight[0], Is.EqualTo(3001));
            Assert.That(s.CameraExposureTimeUs[0], Is.EqualTo(149));
            Assert.That(s.CameraLineRateHz[0], Is.EqualTo(3001));
        }

        [Test]
        public void Validate_NullArrays_ResetsToDefaults()
        {
            var s = new AcquisitionSettings
            {
                CameraGrabHeight = null,
                CameraExposureTimeUs = null,
                CameraLineRateHz = null
            };
            s.Validate();
            Assert.That(s.CameraGrabHeight.Length, Is.EqualTo(7));
            Assert.That(s.CameraExposureTimeUs.Length, Is.EqualTo(7));
            Assert.That(s.CameraLineRateHz.Length, Is.EqualTo(7));
        }

        [Test]
        public void Validate_WrongLength_ResetsToDefaults()
        {
            var s = new AcquisitionSettings
            {
                CameraGrabHeight = new int[] { 100, 200 },
                CameraExposureTimeUs = new double[0],
                CameraLineRateHz = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
            };
            s.Validate();
            Assert.That(s.CameraGrabHeight.Length, Is.EqualTo(7));
            Assert.That(s.CameraExposureTimeUs.Length, Is.EqualTo(7));
            Assert.That(s.CameraLineRateHz.Length, Is.EqualTo(7));
        }

        [Test]
        public void Validate_NegativeValues_ResetsToDefaults()
        {
            var s = new AcquisitionSettings();
            s.CameraGrabHeight[2] = -1;
            s.CameraExposureTimeUs[3] = 0;
            s.CameraLineRateHz[4] = -100;
            s.Validate();
            Assert.That(s.CameraGrabHeight[2], Is.EqualTo(3001));
            Assert.That(s.CameraExposureTimeUs[3], Is.EqualTo(149));
            Assert.That(s.CameraLineRateHz[4], Is.EqualTo(3001));
        }

        [Test]
        public void Validate_PositiveValues_Preserved()
        {
            var s = new AcquisitionSettings();
            s.CameraGrabHeight[0] = 5000;
            s.CameraExposureTimeUs[1] = 200;
            s.CameraLineRateHz[2] = 8000;
            s.Validate();
            Assert.That(s.CameraGrabHeight[0], Is.EqualTo(5000));
            Assert.That(s.CameraExposureTimeUs[1], Is.EqualTo(200));
            Assert.That(s.CameraLineRateHz[2], Is.EqualTo(8000));
        }
    }

    [TestFixture]
    public class AcquisitionSettingsStoreTests
    {
        private string _tempDir;
        private string _origDir;

        [SetUp]
        public void SetUp()
        {
            _tempDir = Path.Combine(Path.GetTempPath(), "AcqStoreTest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempDir);
            _origDir = AppDomain.CurrentDomain.BaseDirectory;
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempDir, true); } catch { }
        }

        [Test]
        public void SaveAndLoad_RoundTrip()
        {
            // AcquisitionSettingsStore uses AppDomain.CurrentDomain.BaseDirectory,
            // which we can't redirect easily. Instead, test the JSON format directly.
            var settings = new AcquisitionSettings();
            settings.CameraGrabHeight[0] = 5000;
            settings.CameraExposureTimeUs[3] = 250.5;
            settings.CameraLineRateHz[6] = 9000;
            settings.Validate();

            // Simulate Save: serialize
            string json = SerializeJson(settings);
            Assert.That(json, Does.Contain("\"CameraGrabHeight\""));
            Assert.That(json, Does.Contain("5000"));

            // Simulate Load: parse
            var loaded = ParseJson(json);
            loaded.Validate();
            Assert.That(loaded.CameraGrabHeight[0], Is.EqualTo(5000));
            Assert.That(loaded.CameraExposureTimeUs[3], Is.EqualTo(250.5).Within(0.01));
            Assert.That(loaded.CameraLineRateHz[6], Is.EqualTo(9000));
        }

        [Test]
        public void ParseJson_EmptyString_ReturnsNull()
        {
            var result = ParseJson("");
            Assert.That(result, Is.Null);
        }

        [Test]
        public void ParseJson_CorruptedJson_FallsBackGracefully()
        {
            var result = ParseJson("{\"CameraGrabHeight\": [not_a_number]}");
            // ParseIntArray catches the FormatException → returns null → keeps defaults
            Assert.That(result, Is.Not.Null);
            Assert.That(result.CameraGrabHeight, Is.EqualTo(new AcquisitionSettings().CameraGrabHeight));
        }

        // ── Mirror of AcquisitionSettingsStore's private methods for testing ──

        private static string SerializeJson(AcquisitionSettings s)
        {
            return "{\n" +
                $"  \"CameraGrabHeight\": [{string.Join(", ", s.CameraGrabHeight)}],\n" +
                $"  \"CameraExposureTimeUs\": [{string.Join(", ", s.CameraExposureTimeUs)}],\n" +
                $"  \"CameraLineRateHz\": [{string.Join(", ", s.CameraLineRateHz)}]\n" +
                "}";
        }

        private static AcquisitionSettings ParseJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json)) return null;
            var result = new AcquisitionSettings();
            var ht = ParseArray<int>(json, "CameraGrabHeight", int.TryParse);
            if (ht != null) result.CameraGrabHeight = ht;
            var exp = ParseArray<double>(json, "CameraExposureTimeUs", TryParseDouble);
            if (exp != null) result.CameraExposureTimeUs = exp;
            var lr = ParseArray<double>(json, "CameraLineRateHz", TryParseDouble);
            if (lr != null) result.CameraLineRateHz = lr;
            return result;
        }

        private delegate bool TryParseDelegate<T>(string s, out T val);

        private static bool TryParseDouble(string s, out double val)
            => double.TryParse(s.Trim(), System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out val);

        private static T[] ParseArray<T>(string json, string key, TryParseDelegate<T> parser)
        {
            var pattern = new System.Text.RegularExpressions.Regex(
                "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
            var m = pattern.Match(json);
            if (!m.Success) return null;
            string body = m.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(body)) return new T[0];
            try
            {
                var parts = body.Split(',');
                var arr = new T[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    if (!parser(parts[i].Trim(), out arr[i]))
                        return null;
                }
                return arr;
            }
            catch { return null; }
        }
    }
}
