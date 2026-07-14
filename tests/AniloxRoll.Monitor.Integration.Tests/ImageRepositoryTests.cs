using System;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class ImageRepositoryTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "ImageRepositoryTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void LoadDirectory_BuildsSortedDistinctPeriodIndexAndReplacesItOnReload()
        {
            WriteRaw("20260714_120001.250-2_raw.jpg");
            WriteRaw("20260714_120000.125-1_raw.jpg");
            WriteRaw("20260714_120000.125-2_raw.jpg");

            var repository = new ImageRepository();
            repository.LoadDirectory(_tempRoot);

            Assert.That(repository.GetAvailablePeriods(), Is.EqualTo(new[]
            {
                new DateTime(2026, 7, 14, 12, 0, 0, 125),
                new DateTime(2026, 7, 14, 12, 0, 1, 250),
            }));

            string secondRoot = Path.Combine(_tempRoot, "reload");
            Directory.CreateDirectory(secondRoot);
            File.WriteAllText(Path.Combine(secondRoot, "20260715_010203.004-1_raw.jpg"), "raw");
            repository.LoadDirectory(secondRoot);

            Assert.That(repository.GetAvailablePeriods(), Is.EqualTo(new[]
            {
                new DateTime(2026, 7, 15, 1, 2, 3, 4),
            }));
        }

        private void WriteRaw(string fileName)
        {
            File.WriteAllText(Path.Combine(_tempRoot, fileName), "raw");
        }
    }
}
