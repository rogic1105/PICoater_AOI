using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    internal static class CursorStatusTextFormatter
    {
        public static string Format(ImageDisplayView.CursorStatus status, string tag = null)
        {
            if (status == null) return string.Empty;

            string prefix = string.IsNullOrWhiteSpace(tag) ? string.Empty : $"影像 [{tag}] | ";
            string mag = status.PhysMag > 0 ? $"{status.PhysMag:F2}x" : "-";

            return prefix +
                   $"位置:({status.CurMmX:F2}, {status.CurMmY:F2}) mm | " +
                   $"X範圍:{status.ViewLeftMm:F1}~{status.ViewRightMm:F1} mm | " +
                   $"Y範圍:{status.ViewTopMm:F1}~{status.ViewBotMm:F1} mm | " +
                   $"座標: ({status.CursorX}, {status.CursorY}) | " +
                   $"亮度: {status.Brightness} | " +
                   $"實體倍率:{mag}";
        }
    }
}
