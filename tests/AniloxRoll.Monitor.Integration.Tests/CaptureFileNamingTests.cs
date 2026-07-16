using System;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class CaptureFileNamingTests
    {
        private string _root;

        [SetUp]
        public void SetUp()
        {
            _root = Path.Combine(Path.GetTempPath(), "CaptureFileNaming_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_root);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_root, true); } catch { }
        }

        [TestCase("c", "_proc_c.jpg")]
        [TestCase("r", "_proc_r.jpg")]
        public void ResolveProcJpg_CurrentAxisNames_ArePreferred(string axis, string suffix)
        {
            string basePath = Path.Combine(_root, "capture");
            string expected = basePath + suffix;
            File.WriteAllText(expected, "current");

            Assert.That(CaptureFileNaming.ResolveProcJpg(basePath, axis), Is.EqualTo(expected));
        }

        [TestCase("c", "_proc_v.jpg")]
        [TestCase("r", "_proc_h.jpg")]
        public void ResolveProcJpg_LegacyAxisNames_RemainReadable(string axis, string suffix)
        {
            string basePath = Path.Combine(_root, "capture");
            string expected = basePath + suffix;
            File.WriteAllText(expected, "legacy");

            Assert.That(CaptureFileNaming.ResolveProcJpg(basePath, axis), Is.EqualTo(expected));
        }
    }
}
