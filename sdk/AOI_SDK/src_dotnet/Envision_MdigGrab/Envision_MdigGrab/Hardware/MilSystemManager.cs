using Matrox.MatroxImagingLibrary;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 管理 MIL Application Context 與各張擷取卡（System）的分配與釋放。
    /// 使用 MappAlloc（而非 MappAllocDefault）避免 MIL 自動分配預設 System，
    /// 以確保多卡場景下能手動控制每張卡的初始化順序。
    /// </summary>
    public static class MilSystemManager
    {
        /// <summary>目前已分配的 MIL Application ID。未初始化時為 M_NULL。</summary>
        public static MIL_ID MilApplication = MIL.M_NULL;

        private static bool _isInitialized = false;

        /// <summary>
        /// 初始化 MIL Application Context。
        /// 重複呼叫時無效（僅初始化一次）。
        /// 同時停用 MIL 內建的錯誤彈窗，改由程式內部處理錯誤。
        /// </summary>
        public static void Initialize()
        {
            if (_isInitialized) return;

            // 使用 MappAlloc 只分配 Application Context，不自動開啟任何 System
            MIL.MappAlloc(MIL.M_NULL, MIL.M_DEFAULT, ref MilApplication);

            // 停用預設錯誤視窗，改由程式自行判斷錯誤碼
            MIL.MappControl(MIL.M_DEFAULT, MIL.M_ERROR, MIL.M_PRINT_DISABLE);

            _isInitialized = true;
        }

        /// <summary>
        /// 分配指定的擷取卡（System）。
        /// 每張實體卡片獨立呼叫一次，以 systemNum 區分不同卡片。
        /// </summary>
        /// <param name="systemDescriptor">System 驅動描述字串，例如 MIL.M_SYSTEM_RADIENTEVCL</param>
        /// <param name="systemNum">卡片編號（0 = 第一張卡，1 = 第二張卡，以此類推）</param>
        /// <returns>分配成功的 System ID；失敗時回傳 M_NULL</returns>
        public static MIL_ID AllocateSystem(string systemDescriptor, int systemNum)
        {
            if (MilApplication == MIL.M_NULL) return MIL.M_NULL;

            MIL_ID sysId = MIL.M_NULL;
            MIL.MsysAlloc(MilApplication, systemDescriptor, systemNum, MIL.M_DEFAULT, ref sysId);
            return sysId;
        }

        /// <summary>
        /// 釋放指定的 MIL System（擷取卡資源）。
        /// 傳入 M_NULL 時不執行任何動作。
        /// </summary>
        /// <param name="sysId">要釋放的 System ID</param>
        public static void FreeSystem(MIL_ID sysId)
        {
            if (sysId != MIL.M_NULL)
                MIL.MsysFree(sysId);
        }

        /// <summary>
        /// 釋放 MIL Application Context，重置初始化狀態。
        /// 必須在所有 System 和 Digitizer 已釋放後才能呼叫。
        /// </summary>
        public static void FreeApplication()
        {
            if (MilApplication != MIL.M_NULL)
            {
                MIL.MappFreeDefault(MilApplication, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
                MilApplication = MIL.M_NULL;
                _isInitialized = false;
            }
        }
    }
}
