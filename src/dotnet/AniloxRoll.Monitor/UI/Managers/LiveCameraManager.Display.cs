using System;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.UI.Managers
{
    // partial：顯示職責已提取到 LiveDisplayCoordinator。
    // 本檔保留既有對外 API，讓 AniloxRollForm 呼叫端不需要改。
    public partial class LiveCameraManager
    {
        public int SelectedMainCameraId => _display.SelectedMainCameraId;

        public event Action<double, double, double, double> OnLiveViewRange
        {
            add { _display.OnLiveViewRange += value; }
            remove { _display.OnLiveViewRange -= value; }
        }

        public Action OnAfterVerticalZoom
        {
            get { return _display.OnAfterVerticalZoom; }
            set { _display.OnAfterVerticalZoom = value; }
        }

        public double RowPitchMm
        {
            get { return _display.RowPitchMm; }
            set { _display.RowPitchMm = value; }
        }

        public void RefreshMainDisplay()
        {
            _display.SwitchMainDisplay(_display.SelectedMainCameraId);
        }

        public void ResetMainDisplayView()
        {
            _display.ResetMainDisplayView();
        }

        public void ApplyMainDisplayMode()
        {
            _display.ApplyMainDisplayMode();
        }

        public void RefreshWaterfallDisplay()
        {
            _display.RefreshWaterfallDisplay();
        }

        public void TeardownWaterfallDisplay()
        {
            _display.TeardownWaterfallDisplay();
        }

        // ── 背景預覽（共用顯示路 forwarder；政策見 AniloxRollForm.Background.cs）──
        public void EnterBackgroundPreview() => _display.EnterBackgroundPreview();
        public void ExitBackgroundPreview() => _display.ExitBackgroundPreview(ExpectedCameraCount > 0 ? ExpectedCameraCount : 7);
        public void PushStaticFrame(int camId, byte[] gray, int w, int h) => _display.PushStaticFrame(camId, gray, w, h);

        public void ClearCameraFrame(int camId)
        {
            _display.ClearCameraFrame(camId);
        }

        public void ApplyDisplayDirection()
        {
            _display.ApplyDisplayDirection();
        }

        public void SetLodMode(LiveLodMode mode)
        {
            _display.SetLodMode(mode);
        }

        public void SetPhysicalMagnification1x()
        {
            _display.SetPhysicalMagnification1x();
        }

        internal void ApplyCustomZoom(int wheelDelta)
        {
            _display.ApplyCustomZoom(wheelDelta);
        }
    }
}
