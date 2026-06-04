namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 相機硬體/擷取卡配置（偏靜態，通常於安裝時決定）。
    /// </summary>
    public class CameraHardwareConfig
    {
        public int Id { get; set; }
        public string SystemDescriptor { get; set; }
        public int SystemNum { get; set; }

        /// <summary>板內 device number（0-based，== MIL.M_DEVx，M_DEV0=0）。
        /// 必須是 int 不可用 MIL_INT：JavaScriptSerializer 無法序列化 MIL_INT struct（會寫成 {}），
        /// 導致缺檔重生的 system-settings.json 失效（DevNum 全變 0 → 只認到單一 board）。
        /// 消費端 MdigAlloc 需要時自行 (MIL_INT) 轉換。</summary>
        public int DevNum { get; set; }
        public string DcfPath { get; set; }
    }
}
