using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Core.Camera
{
    public static class CameraSystemManager
    {
        private static MIL_ID _milApplication = MIL.M_NULL;
        private static bool _isInitialized = false;

        public static MIL_ID MilApplication
        {
            get { return _milApplication; }
        }

        public static void Initialize()
        {
            if (_isInitialized) return;

            // [修正] 改用 MappAlloc，只分配 Application Context
            // 原因：MappAllocDefault 會自動分配預設 System，這會干擾我們後續手動分配多張卡的操作
            // 參數 1: Server Description (本機使用 M_NULL)
            // 參數 2: Init Flag (M_DEFAULT)
            // 參數 3: Application ID (接收分配結果)
            MIL.MappAlloc(MIL.M_NULL, MIL.M_DEFAULT, ref _milApplication);

            // 禁用預設錯誤視窗 (改由程式內部處理)
            MIL.MappControl(MIL.M_DEFAULT, MIL.M_ERROR, MIL.M_PRINT_DISABLE);

            _isInitialized = true;
        }

        public static MIL_ID AllocateSystem(string systemDescriptor, int systemNum)
        {
            if (_milApplication == MIL.M_NULL) return MIL.M_NULL;

            MIL_ID sysId = MIL.M_NULL;

            // 這裡才是真正分配硬體資源 (擷取卡) 的地方
            MIL.MsysAlloc(_milApplication, systemDescriptor, systemNum, MIL.M_DEFAULT, ref sysId);

            return sysId;
        }

        public static void FreeSystem(MIL_ID sysId)
        {
            if (sysId != MIL.M_NULL)
            {
                MIL.MsysFree(sysId);
            }
        }

        public static void FreeApplication()
        {
            if (_milApplication != MIL.M_NULL)
            {
                MIL.MappFreeDefault(_milApplication, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
                _milApplication = MIL.M_NULL;
                _isInitialized = false;
            }
        }
    }
}
