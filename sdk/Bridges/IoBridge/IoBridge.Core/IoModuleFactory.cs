using System;

namespace IoBridge.Core
{
    /// <summary>
    /// 依型號建對應 IO module client。型號 → 實作的單一決策點。
    ///
    /// 新增型號兩種情況：
    ///   1. 同廠商不同型號（同 protocol，只差 DI/DO 點數）→ 沿用現有 client，加 case 即可
    ///      （未來點數差異可在此傳參數給 client）
    ///   2. 換廠商 / protocol（如 Advantech ADAM）→ 在 Modules/ 寫新 IModbusTcpClient 實作，
    ///      加 case 對應
    ///
    /// Modules/ 資料夾按「廠商 / protocol」分（ls 一眼看支援哪些）；本 factory 按「型號」
    /// map 到對應實作。
    /// </summary>
    public static class IoModuleFactory
    {
        /// <summary>目前支援的型號清單（PG 下拉 / 驗證用）。</summary>
        public static readonly string[] SupportedModels = { "ET-7044" };

        public static IModbusTcpClient Create(string model)
        {
            switch ((model ?? "").Trim().ToUpperInvariant())
            {
                case "":          // 空 = 預設
                case "ET-7044":
                    return new IcpDasModbusTcpClient();   // ICP DAS 標準 Modbus（ET 系列通用）
                // 範例（未來擴充）：
                // case "ADAM-6050": return new AdvantechAdamClient();
                default:
                    throw new NotSupportedException(
                        $"未支援的 IO 型號: {model}（目前支援: {string.Join(", ", SupportedModels)}）");
            }
        }
    }
}
