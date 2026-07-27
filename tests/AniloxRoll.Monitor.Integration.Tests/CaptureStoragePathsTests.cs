using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class CaptureStoragePathsTests
    {
        private const string ConfiguredRoot = @"D:\Anilox\Captures";

        [TestCase(@"D:\Anilox\Captures_pack", ConfiguredRoot)]
        [TestCase(@"D:\Anilox", ConfiguredRoot)]
        [TestCase(@"D:\Anilox\Captures", ConfiguredRoot)]
        public void ResolveSelectedDataRoot_ProductRoots_ReturnsConfiguredRoot(
            string selectedRoot,
            string expected)
        {
            Assert.That(
                CaptureStoragePaths.ResolveSelectedDataRoot(
                    selectedRoot,
                    ConfiguredRoot),
                Is.EqualTo(expected).IgnoreCase);
        }

        [Test]
        public void ResolveSelectedDataRoot_ExternalArchive_PreservesSelection()
        {
            const string externalRoot = @"E:\ArchivedRuns";

            Assert.That(
                CaptureStoragePaths.ResolveSelectedDataRoot(
                    externalRoot,
                    ConfiguredRoot),
                Is.EqualTo(externalRoot));
        }

        [TestCase(
            @"D:\Anilox\Captures_pack",
            @"D:\Anilox\Captures")]
        [TestCase(
            @"\\192.168.10.20\Anilox\Captures_pack",
            @"\\192.168.10.20\Anilox\Captures")]
        public void UpgradeLegacyPackedRoot_ReplacesOnlyFinalDirectory(
            string legacyRoot,
            string expected)
        {
            Assert.That(
                CaptureStoragePaths.UpgradeLegacyPackedRoot(legacyRoot),
                Is.EqualTo(expected).IgnoreCase);
        }
    }
}
