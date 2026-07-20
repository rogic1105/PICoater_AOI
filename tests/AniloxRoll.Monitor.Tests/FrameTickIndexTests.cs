using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class FrameTickIndexTests
    {
        [Test]
        public void ResolveAlignment_AllTicksPresent_UsesTickSlotsAndPreservesMissingFrame()
        {
            const string cam1First = @"C:\captures\260721-080000000-1_raw.jpg";
            const string cam1Second = @"C:\captures\260721-080000100-1_raw.jpg";
            const string cam2First = @"C:\captures\260721-080000000-2_raw.jpg";
            var grouped = new Dictionary<int, List<string>>
            {
                [1] = new List<string> { cam1First, cam1Second },
                [2] = new List<string> { cam2First }
            };
            var ticks = new Dictionary<string, long>
            {
                ["260721-080000000-1"] = 1000,
                ["260721-080000100-1"] = 2000,
                ["260721-080000000-2"] = 1005
            };

            FrameAlignmentResult result = FrameTickIndex.ResolveAlignment(grouped, ticks);

            Assert.That(result.UsedHardwareTicks, Is.True);
            Assert.That(result.Mode, Is.EqualTo("tick"));
            Assert.That(result.AllPaths, Has.Count.EqualTo(3));
            Assert.That(result.ByCamera[1], Is.EqualTo(new[] { cam1First, cam1Second }));
            Assert.That(result.ByCamera[2], Is.EqualTo(new[] { cam2First, null }));
        }

        [Test]
        public void ResolveAlignment_AnyTickMissing_FallsBackToFilenameSlots()
        {
            const string cam1First = @"C:\captures\260721-080000000-1_raw.jpg";
            const string cam1Second = @"C:\captures\260721-080000100-1_raw.jpg";
            const string cam2First = @"C:\captures\260721-080000000-2_raw.jpg";
            var grouped = new Dictionary<int, List<string>>
            {
                [1] = new List<string> { cam1First, cam1Second },
                [2] = new List<string> { cam2First }
            };
            var incompleteTicks = new Dictionary<string, long>
            {
                ["260721-080000000-1"] = 1000,
                ["260721-080000100-1"] = 2000
            };

            FrameAlignmentResult result = FrameTickIndex.ResolveAlignment(grouped, incompleteTicks);

            Assert.That(result.UsedHardwareTicks, Is.False);
            Assert.That(result.Mode, Is.EqualTo("filename"));
            Assert.That(result.ByCamera[1], Is.EqualTo(new[] { cam1First, cam1Second }));
            Assert.That(result.ByCamera[2], Is.EqualTo(new[] { cam2First, null }));
        }
    }
}
