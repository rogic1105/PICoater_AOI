using System;
using System.Collections.Generic;
using System.Drawing;
using System.Diagnostics;

namespace AniloxRoll.Monitor.UI.Services
{
    /// <summary>
    /// Owns the lifetime of processed review bitmaps that are not displayed directly.
    /// </summary>
    public sealed class ImageCacheService
    {
        private readonly List<Image> _images = new List<Image>();

        public void Track(Image image)
        {
            if (image != null) _images.Add(image);
        }

        public void Clear()
        {
            foreach (var image in _images)
            {
                try { image.Dispose(); }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[ImageCacheService] Bitmap.Dispose failed: {ex.GetType().Name}: {ex.Message}");
                }
            }

            _images.Clear();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, blocking: false);
        }
    }
}
