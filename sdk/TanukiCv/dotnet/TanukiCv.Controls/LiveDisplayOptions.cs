using System.Drawing;
using TanukiCv.Core;

namespace TanukiCv.Controls
{
    /// <summary>
    /// Shared display switches for <see cref="ImageDisplayView"/> callers.
    /// Frame feeding, layout, and LOD provider selection remain caller-owned.
    /// </summary>
    public sealed class LiveDisplayOptions
    {
        public bool MergeMode { get; set; }
        public bool MergeAll { get; set; }
        public MergeOverlap MergeStrategy { get; set; } = MergeOverlap.Midline;
        public bool FlipVertical { get; set; }
        public Color? ThumbSelectedColor { get; set; }
    }
}
