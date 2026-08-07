using System;
using AniloxRoll.Monitor.Core.Data;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    // partial：顯示職責已提取到 LiveDisplayCoordinator。
    // 本檔保留既有對外 API，讓 AniloxRollForm 呼叫端不需要改。
    public partial class LiveCameraManager
    {
        public int SelectedMainCameraId => _display.SelectedMainCameraId;

        public CanvasOverlayMode CanvasOverlayMode
        {
            get { return _display.OverlayMode; }
            set { _display.OverlayMode = value; }
        }

        public event Action<CanvasOverlayMode> OnCanvasOverlayModeChanged
        {
            add { _display.OverlayModeChanged += value; }
            remove { _display.OverlayModeChanged -= value; }
        }

        public event Action<double, double, double, double> OnLiveViewRange
        {
            add { _display.OnLiveViewRange += value; }
            remove { _display.OnLiveViewRange -= value; }
        }

        public double RowPitchMm
        {
            get { return _display.RowPitchMm; }
            set { _display.RowPitchMm = value; }
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
        public bool IsBgPreviewActive => _display.IsBgPreviewActive;
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

        public void RefreshEnhanceColorMap()
        {
            _display.RefreshEnhanceColorMap();
        }

        public void RefreshEnhanceIntensityScales(float columnScale, float rowScale)
        {
            _display.SetEnhanceIntensityScales(columnScale, rowScale);
        }

        public void RefreshHorizontalDisplayCrop()
        {
            _display.RefreshHorizontalDisplayCrop();
        }

        public void RefreshHorizontalDisplayCrop(double trimHeadMm, double trimTailMm)
        {
            _display.RefreshHorizontalDisplayCrop(trimHeadMm, trimTailMm);
        }

        public void SetLodMode(LiveLodMode mode)
        {
            _display.SetLodMode(mode);
        }
    }
}
