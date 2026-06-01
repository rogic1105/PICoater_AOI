using System.Linq;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// AcquisitionSettings 預設值集中定義（同 InspectionDefaults 風格）。
    /// 原本 3001/149/3001 在 AcquisitionSettings.cs 寫 5 處（屬性初始 + Validate 兩處），
    /// 收斂到唯一來源避免改一處漏改其他。
    /// </summary>
    internal static class AcquisitionDefaults
    {
        public const int    CamCount       = 7;
        public const int    GrabHeight     = 3001;
        public const double ExposureTimeUs = 50.0;     // PICoater 機台實測
        public const double LineRateHz     = 3001.0;

        public static int[]    NewGrabHeightArray()   => Enumerable.Repeat(GrabHeight,     CamCount).ToArray();
        public static double[] NewExposureTimeArray() => Enumerable.Repeat(ExposureTimeUs, CamCount).ToArray();
        public static double[] NewLineRateArray()     => Enumerable.Repeat(LineRateHz,     CamCount).ToArray();
    }
}
