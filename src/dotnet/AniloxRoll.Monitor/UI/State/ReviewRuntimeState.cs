using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.State
{
    /// <summary>
    /// Runtime-only review metadata shared by loaders and display coordination.
    /// </summary>
    public sealed class ReviewRuntimeState
    {
        public CsvConfigSnapshot Config { get; set; }
        public double ScreenMmPerPixel { get; set; }
    }
}
