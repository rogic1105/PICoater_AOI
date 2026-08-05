using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReviewAdjacentPrefetchTests
    {
        [Test]
        public void Select_MovingTowardLargerIndex_PrioritizesThatNeighbor()
        {
            List<GrabIdInfo> items = BuildItems(5);

            GrabIdInfo[] selected = ReviewAdjacentPrefetchPolicy.Select(
                items, currentIndex: 2, direction: 1);

            Assert.That(selected.Length, Is.EqualTo(2));
            Assert.That(selected[0].GrabId, Is.EqualTo("id3"));
            Assert.That(selected[1].GrabId, Is.EqualTo("id1"));
        }

        [Test]
        public void Select_AtEdge_ReturnsOnlyExistingNeighbor()
        {
            List<GrabIdInfo> items = BuildItems(3);

            GrabIdInfo[] selected = ReviewAdjacentPrefetchPolicy.Select(
                items, currentIndex: 0, direction: -1);

            Assert.That(selected.Length, Is.EqualTo(1));
            Assert.That(selected[0].GrabId, Is.EqualTo("id1"));
        }

        [Test]
        public async Task Cache_SecondReadIsHit_AndOldestEntryIsEvicted()
        {
            using (var cache = new ReviewAsyncLruCache<string>(
                maxEntries: 1, maxSize: 32, sizeOf: value => value.Length))
            {
                ReviewCacheAccess firstAccess;
                string first = await cache.GetOrLoadAsync(
                    "a", () => "alpha", out firstAccess);
                ReviewCacheAccess hitAccess;
                string hit = await cache.GetOrLoadAsync(
                    "a", () => "unexpected", out hitAccess);

                Assert.That(first, Is.EqualTo("alpha"));
                Assert.That(hit, Is.EqualTo("alpha"));
                Assert.That(firstAccess, Is.EqualTo(ReviewCacheAccess.Cold));
                Assert.That(hitAccess, Is.EqualTo(ReviewCacheAccess.Hit));

                ReviewCacheAccess secondAccess;
                await cache.GetOrLoadAsync("b", () => "beta", out secondAccess);

                Assert.That(secondAccess, Is.EqualTo(ReviewCacheAccess.Cold));
                Assert.That(cache.Count, Is.EqualTo(1));
                Assert.That(cache.TryGet("a", out _), Is.False);
                Assert.That(cache.TryGet("b", out string beta), Is.True);
                Assert.That(beta, Is.EqualTo("beta"));
            }
        }

        [Test]
        public async Task Cache_NullLoad_IsNotStoredAsAHit()
        {
            using (var cache = new ReviewAsyncLruCache<string>(
                maxEntries: 2, maxSize: 32, sizeOf: value => value.Length))
            {
                ReviewCacheAccess firstAccess;
                string missing = await cache.GetOrLoadAsync(
                    "missing", () => null, out firstAccess);
                ReviewCacheAccess secondAccess;
                string loaded = await cache.GetOrLoadAsync(
                    "missing", () => "available", out secondAccess);

                Assert.That(missing, Is.Null);
                Assert.That(firstAccess, Is.EqualTo(ReviewCacheAccess.Cold));
                Assert.That(loaded, Is.EqualTo("available"));
                Assert.That(secondAccess, Is.EqualTo(ReviewCacheAccess.Cold));
                Assert.That(cache.Count, Is.EqualTo(1));
            }
        }

        private static List<GrabIdInfo> BuildItems(int count)
        {
            var items = new List<GrabIdInfo>(count);
            for (int i = 0; i < count; i++)
            {
                items.Add(new GrabIdInfo
                {
                    GrabId = "id" + i,
                    Earliest = new DateTime(2026, 8, 4).AddSeconds(i),
                    Latest = new DateTime(2026, 8, 4).AddSeconds(i + 1)
                });
            }
            return items;
        }
    }
}
