using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 單台相機的靜態設定資料。
    /// 在 GrabForm 建構子中填入，傳遞給 CameraSession.Initialize() 使用。
    /// </summary>
    public class CameraConfig
    {
        /// <summary>相機識別 ID，用於 ListView 顯示與事件對應</summary>
        public int Id { get; set; }

        /// <summary>擷取卡驅動描述字串，例如 MIL.M_SYSTEM_RADIENTEVCL</summary>
        public string SystemDescriptor { get; set; }

        /// <summary>擷取卡編號（0 = 第一張卡，1 = 第二張卡）</summary>
        public int SystemNum { get; set; }

        /// <summary>該擷取卡上的 Digitizer Device 編號（通常為 M_DEV0）</summary>
        public MIL_INT DevNum { get; set; }

        /// <summary>DCF 設定檔完整路徑</summary>
        public string DcfPath { get; set; }

        /// <summary>
        /// 手動填入的曝光設定值（μs）。
        /// Camera Link 無 CLProtocol 時無法從硬體讀回曝光，需在此處填入與 Intellicam 一致的數值。
        /// </summary>
        public double ExposureUs { get; set; } = 0;

        /// <summary>用於顯示 Live 影像的 Panel 控制項</summary>
        public Panel DisplayPanel { get; set; }

        /// <summary>用於顯示相機連線狀態（Online / Offline）的 Label 控制項</summary>
        public Label StatusLabel { get; set; }
    }
}
