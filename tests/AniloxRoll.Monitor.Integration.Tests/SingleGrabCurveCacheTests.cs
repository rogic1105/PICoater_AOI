using System.Threading;
using NUnit.Framework;
using AniloxRoll.Monitor.UI.Services;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class SingleGrabCurveCacheTests
    {
        [Test]
        public void GetOrLoadAsync_SameInflightKey_LoadsOnce()
        {
            using (var cache = new SingleGrabCurveCache(8, 1024 * 1024))
            using (var release = new ManualResetEventSlim(false))
            using (var started = new ManualResetEventSlim(false))
            {
                int loads = 0;
                var first = cache.GetOrLoadAsync("grab-a", () =>
                {
                    Interlocked.Increment(ref loads);
                    started.Set();
                    release.Wait();
                    return Profile(1f);
                });
                Assert.That(started.Wait(1000), Is.True);

                var second = cache.GetOrLoadAsync("grab-a", () =>
                {
                    Interlocked.Increment(ref loads);
                    return Profile(2f);
                });
                release.Set();

                Assert.That(first.GetAwaiter().GetResult(), Is.SameAs(second.GetAwaiter().GetResult()));
                Assert.That(loads, Is.EqualTo(1));
                Assert.That(cache.Count, Is.EqualTo(1));
            }
        }

        [Test]
        public void CapacityExceeded_EvictsLeastRecentlyUsedProfile()
        {
            using (var cache = new SingleGrabCurveCache(8, 24))
            {
                cache.GetOrLoadAsync("grab-a", () => Profile(1f)).GetAwaiter().GetResult();
                cache.GetOrLoadAsync("grab-b", () => Profile(2f)).GetAwaiter().GetResult();

                Assert.That(cache.TryGet("grab-a", out _), Is.False);
                Assert.That(cache.TryGet("grab-b", out var current), Is.True);
                Assert.That(current.Mean[0][0], Is.EqualTo(2f));
            }
        }

        [Test]
        public void Clear_OldInflightLoadCannotOverwriteNewGeneration()
        {
            using (var cache = new SingleGrabCurveCache(8, 1024 * 1024))
            using (var release = new ManualResetEventSlim(false))
            using (var started = new ManualResetEventSlim(false))
            {
                var old = cache.GetOrLoadAsync("grab-a", () =>
                {
                    started.Set();
                    release.Wait();
                    return Profile(1f);
                });
                Assert.That(started.Wait(1000), Is.True);

                cache.Clear();
                var fresh = cache.GetOrLoadAsync("grab-a", () => Profile(2f));
                Assert.That(fresh.GetAwaiter().GetResult().Mean[0][0], Is.EqualTo(2f));
                release.Set();
                old.GetAwaiter().GetResult();

                Assert.That(cache.TryGet("grab-a", out var current), Is.True);
                Assert.That(current.Mean[0][0], Is.EqualTo(2f));
            }
        }

        private static SingleGrabCurveProfile Profile(float value)
        {
            return new SingleGrabCurveProfile(
                new[] { new[] { value, value } },
                new[] { new[] { value, value } },
                1, "bins", 0, 0, 0);
        }
    }
}
