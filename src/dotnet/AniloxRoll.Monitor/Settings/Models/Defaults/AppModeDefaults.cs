namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>AppModeConfig 預設值集中定義（與 InspectionDefaults / AcquisitionDefaults / SystemDefaults 同風格）。</summary>
    internal static class AppModeDefaults
    {
        public const MachineRole Role              = MachineRole.Inspection;
        public const string      StorageMachineConfigFolder = @"D:\Anilox\Config";
        public const string      StorageMachineDataPath = "";
        public const int         StorageMinFreeGB = 100;
    }
}
