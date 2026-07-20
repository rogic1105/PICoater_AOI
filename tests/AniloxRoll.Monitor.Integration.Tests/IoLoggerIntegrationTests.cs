using System;
using System.IO;
using IoBridge.Core;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    [NonParallelizable]
    public class IoLoggerIntegrationTests
    {
        [Test]
        public void Write_WithInjectedDirectory_UsesManagedFileName()
        {
            string root = Path.Combine(Path.GetTempPath(), "IoLogger_" + Guid.NewGuid().ToString("N"));
            string originalDirectory = IoLogger.LogDirectory;
            string originalPrefix = IoLogger.FilePrefix;
            try
            {
                IoLogger.LogDirectory = root;
                IoLogger.FilePrefix = "io";

                IoLogger.Info("integration probe");

                string expected = Path.Combine(root, "io-" + DateTime.Now.ToString("yyyyMMdd") + ".log");
                Assert.That(File.Exists(expected), Is.True);
                Assert.That(File.ReadAllText(expected), Does.Contain("integration probe"));
            }
            finally
            {
                IoLogger.LogDirectory = originalDirectory;
                IoLogger.FilePrefix = originalPrefix;
                try { Directory.Delete(root, true); } catch { }
            }
        }
    }
}
