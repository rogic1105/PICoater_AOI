namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>已整合至 InspectionSettingsStore，保留以避免外部參考錯誤。</summary>
    internal static class InspectionSettingsDefaultsProvider
    {
        public static InspectionSettings LoadDefaults() => InspectionSettingsStore.Load();
    }
}
