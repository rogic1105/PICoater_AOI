using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 管理多台相機的生命週期：初始化、抓圖開關、釋放資源。
    /// 不含任何 UI 依賴，所有 UI 互動透過 Event 或 callback 回傳給呼叫端。
    /// </summary>
    public class CameraSession
    {
        private readonly List<MilCameraUnit> _cameras = new List<MilCameraUnit>();

        /// <summary>Key = SystemNum，Value = 已分配的 MIL System ID。避免同一張卡重複開啟。</summary>
        private readonly Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

        private bool _userWantsGrab = false;

        /// <summary>正在執行釋放流程時為 true，用於阻止 Timer Tick 繼續查詢相機狀態。</summary>
        public volatile bool IsReleasing = false;

        /// <summary>目前已初始化的相機清單（唯讀）</summary>
        public IReadOnlyList<MilCameraUnit> Cameras => _cameras;

        /// <summary>是否已有相機完成初始化</summary>
        public bool HasCameras => _cameras.Count > 0;

        /// <summary>相機滑鼠座標/像素值變化時觸發，參數：(camId, x, y, pixelValue)</summary>
        public event Action<int, int, int, int> OnMouseDataChanged;

        /// <summary>MIL System 分配失敗時觸發，參數：失敗的 SystemNum</summary>
        public event Action<int> OnSystemAllocFailed;

        // ================= 初始化 =================

        /// <summary>
        /// 依照 configs 逐一分配 MIL System 與 MilCameraUnit，並呼叫各相機的 Initialize()。
        /// 若已有相機存在則直接返回（防止重複初始化）。
        /// System 分配失敗時觸發 OnSystemAllocFailed 並跳過該相機。
        /// </summary>
        /// <param name="configs">相機設定清單</param>
        /// <param name="enableImageProcessing">是否啟用 GPU 影像處理</param>
        public void Initialize(IList<CameraConfig> configs, bool enableImageProcessing)
        {
            if (HasCameras) return;

            IsReleasing = false;
            _userWantsGrab = false;

            MilSystemManager.Initialize();

            foreach (var cfg in configs)
            {
                MIL_ID sysId = MIL.M_NULL;

                // 同一張卡（SystemNum 相同）只分配一次 System
                if (_allocatedSystems.ContainsKey(cfg.SystemNum))
                {
                    sysId = _allocatedSystems[cfg.SystemNum];
                }
                else
                {
                    sysId = MilSystemManager.AllocateSystem(cfg.SystemDescriptor, cfg.SystemNum);
                    if (sysId != MIL.M_NULL)
                        _allocatedSystems.Add(cfg.SystemNum, sysId);
                    else
                    {
                        OnSystemAllocFailed?.Invoke(cfg.SystemNum);
                        continue;
                    }
                }

                var cam = new MilCameraUnit(
                    sysId, cfg.Id, cfg.DevNum, cfg.DcfPath,
                    cfg.DisplayPanel.Handle, enableImageProcessing);

                cam.OnMouseDataChanged += (id, x, y, val) => OnMouseDataChanged?.Invoke(id, x, y, val);
                cam.Initialize();
                cam.SetExposureUs(cfg.ExposureUs);
                _cameras.Add(cam);
            }
        }

        // ================= 抓圖控制 =================

        /// <summary>
        /// 切換所有相機的抓圖狀態（啟動 ↔ 停止）。
        /// 內部記錄 _userWantsGrab 並同步到每台相機的 SetUserGrabIntent()。
        /// </summary>
        public void ToggleGrab()
        {
            _userWantsGrab = !_userWantsGrab;
            foreach (var cam in _cameras)
                cam.SetUserGrabIntent(_userWantsGrab);
        }

        /// <summary>
        /// 即時更新所有相機的影像處理開關。
        /// </summary>
        /// <param name="enabled">true = 啟用 GPU 二值化，false = 直接顯示原圖</param>
        public void SetImageProcessing(bool enabled)
        {
            foreach (var cam in _cameras)
                cam.EnableImageProcessing = enabled;
        }

        /// <summary>
        /// 輪詢每台相機的連線狀態，並在條件符合時自動重新啟動抓圖。
        /// 由 UI 的 Timer Tick 定期呼叫。
        /// </summary>
        /// <param name="onPresenceChanged">
        /// 每台相機的連線狀態回調，由呼叫端（GrabForm）決定如何更新 UI。
        /// 參數：(camId, isConnected)
        /// </param>
        public void UpdatePresence(Action<int, bool> onPresenceChanged)
        {
            foreach (var cam in _cameras)
            {
                bool connected = cam.CheckPresence();
                onPresenceChanged?.Invoke(cam.CameraId, connected);

                // 連線恢復後，若使用者原本希望抓圖則自動重啟
                if (connected && cam.UserWantsGrab && !cam.IsLive)
                    cam.ApplyGrabState();
            }
        }

        // ================= 釋放資源 =================

        /// <summary>
        /// 非同步釋放所有 MIL 資源（相機 → System → Application）。
        /// 供 button3（Free MIL）使用，避免阻塞 UI 執行緒。
        /// </summary>
        public async Task ReleaseAsync()
        {
            IsReleasing = true;
            await Task.Run(() => ReleaseResources());
        }

        /// <summary>
        /// 同步釋放所有 MIL 資源。
        /// 供 OnFormClosing 使用，確保程式關閉前資源完整釋放。
        /// </summary>
        public void Release()
        {
            IsReleasing = true;
            ReleaseResources();
        }

        /// <summary>
        /// 實際執行釋放的私有方法。
        /// 順序：相機（MdigFree）→ System（MsysFree）→ Application（MappFree）。
        /// </summary>
        private void ReleaseResources()
        {
            foreach (var cam in _cameras) cam.Free();
            _cameras.Clear();

            foreach (var kvp in _allocatedSystems) MilSystemManager.FreeSystem(kvp.Value);
            _allocatedSystems.Clear();

            MilSystemManager.FreeApplication();
        }
    }
}
