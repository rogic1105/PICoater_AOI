using System;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public sealed class BackgroundProfileRepositoryTests
    {
        private string _root;
        private BackgroundProfileRepository _repository;

        [SetUp]
        public void SetUp()
        {
            _root = Path.Combine(
                Path.GetTempPath(),
                "PICoater-BackgroundProfileRepository-" + Guid.NewGuid().ToString("N"));
            _repository = new BackgroundProfileRepository(_root);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_root))
                Directory.Delete(_root, true);
        }

        [Test]
        public void SaveAndActivateVersion_RoundTripsMcbfV2AndManifest()
        {
            float[] expected = { 1.25f, 2.5f, 3.75f };

            string path = _repository.SaveCameraProfile(
                expected, expected.Length, 2, "20260728-120000000", 123, 456.5f);
            _repository.ActivateVersion("20260728-120000000");

            BackgroundManifestSnapshot manifest = _repository.ReadManifest();
            Assert.That(manifest.Status, Is.EqualTo(BackgroundManifestStatus.Active));
            Assert.That(manifest.Version, Is.EqualTo("20260728-120000000"));
            Assert.That(
                _repository.ResolveCameraProfilePath(expected.Length, 2),
                Is.EqualTo(path));
            Assert.That(_repository.LoadProfile(path), Is.EqualTo(expected));

            using (var reader = new BinaryReader(File.OpenRead(path)))
            {
                Assert.That(new string(reader.ReadChars(4)), Is.EqualTo("MCBF"));
                Assert.That(reader.ReadInt32(), Is.EqualTo(2));
                Assert.That(reader.ReadSingle(), Is.EqualTo(1.0f));
                Assert.That(reader.ReadInt32(), Is.EqualTo(123));
                Assert.That(reader.ReadSingle(), Is.EqualTo(456.5f));
                Assert.That(reader.ReadInt32(), Is.EqualTo(expected.Length));
            }
        }

        [Test]
        public void InvalidManifest_DoesNotFallBackToLegacyProfile()
        {
            _repository.EnsureDirectory();
            File.WriteAllText(
                Path.Combine(_root, CaptureFileNaming.BgActiveManifest),
                "{ invalid json");
            File.WriteAllBytes(
                Path.Combine(_root, CaptureFileNaming.BgBin(3, 1)),
                new byte[] { 1, 2, 3 });

            BackgroundManifestSnapshot manifest = _repository.ReadManifest();

            Assert.That(manifest.Status, Is.EqualTo(BackgroundManifestStatus.Invalid));
            Assert.That(_repository.HasAnyProfile(), Is.False);
            Assert.That(
                Path.GetFileName(_repository.ResolveCameraProfilePath(3, 1)),
                Is.EqualTo("__invalid-active-background__.bin"));
            Assert.That(_repository.ResolvePreviewProfilePath(1), Is.Null);
        }

        [Test]
        public void CleanupInactiveVersions_KeepsOnlyActiveVersion()
        {
            float[] values = { 1f, 2f, 3f };
            string oldPath = _repository.SaveCameraProfile(
                values, values.Length, 1, "old-version", 10, 20f);
            string activePath = _repository.SaveCameraProfile(
                values, values.Length, 1, "active-version", 10, 20f);
            _repository.ActivateVersion("active-version");

            _repository.CleanupInactiveVersions();

            Assert.That(File.Exists(oldPath), Is.False);
            Assert.That(File.Exists(activePath), Is.True);
            Assert.That(
                _repository.ResolvePreviewProfilePath(1),
                Is.EqualTo(activePath));
        }

        [Test]
        public void DeleteVersion_RemovesOnlyRequestedVersion()
        {
            float[] values = { 1f, 2f };
            string rejectedPath = _repository.SaveCameraProfile(
                values, values.Length, 1, "rejected", 10, 20f);
            string retainedPath = _repository.SaveCameraProfile(
                values, values.Length, 1, "retained", 10, 20f);

            _repository.DeleteVersion("rejected");

            Assert.That(File.Exists(rejectedPath), Is.False);
            Assert.That(File.Exists(retainedPath), Is.True);
        }
    }
}
