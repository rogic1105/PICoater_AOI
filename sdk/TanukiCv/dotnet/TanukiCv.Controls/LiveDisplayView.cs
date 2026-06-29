using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// Backward-compatible name for callers that still reference the old live-specific type.
    /// New code should use <see cref="ImageDisplayView"/>.
    /// </summary>
    public sealed class LiveDisplayView : ImageDisplayView
    {
        public LiveDisplayView(Panel mainPanel, Panel[] camPanels, double screenMmPerPx)
            : base(mainPanel, camPanels, screenMmPerPx)
        {
        }
    }
}
