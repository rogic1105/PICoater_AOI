using System;
using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class NativeBufferPoolTests
    {
        [Test]
        public void Constructor_UsesOneAlignedPinnedSlab_AndFreesItOnce()
        {
            var allocationSizes = new List<ulong>();
            var freedPointers = new List<IntPtr>();
            var slab = new IntPtr(0x10000000);

            using (var pool = new NativeBufferPool(
                16,
                8,
                4,
                size =>
                {
                    allocationSizes.Add(size);
                    return slab;
                },
                ptr => freedPointers.Add(ptr)))
            {
                Assert.That(allocationSizes, Has.Count.EqualTo(1));
                Assert.That(allocationSizes[0], Is.EqualTo(pool.PinnedBytes));

                var pointers = new[]
                {
                    pool.InputBuffer,
                    pool.MuraBuffer,
                    pool.RidgeBuffer,
                    pool.ThumbnailBuffer,
                    pool.CurveMeanBuffer,
                    pool.CurveMaxBuffer,
                    pool.CurveRowMeanBuffer,
                    pool.CurveRowMaxBuffer
                };

                Assert.That(new HashSet<IntPtr>(pointers), Has.Count.EqualTo(pointers.Length));
                foreach (IntPtr pointer in pointers)
                    Assert.That((pointer.ToInt64() - slab.ToInt64()) % 64, Is.EqualTo(0));
            }

            Assert.That(freedPointers, Is.EqualTo(new[] { slab }));
        }
    }
}
