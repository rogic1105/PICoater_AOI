using AniloxRoll.Monitor.Core.Data;
using AOI.SDK.Core.Models;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    /// <summary>
    /// [View Helper] 縮圖畫廊管理器。
    /// 封裝 PictureBox 陣列的顯示邏輯，包含：更新圖片、繪製選取邊框 (Paint)、
    /// 以及處理點擊事件，減輕 Form 的程式碼負擔。
    /// </summary>
    public class ThumbnailGridPresenter
    {
        private PictureBox[] _previewBoxes;
        private int _selectedIndex = 0;

        // 當使用者點擊或程式選取變更時觸發
        public event Action<int> SelectionChanged;

        public int SelectedIndex => _selectedIndex;
        private TimedResult<InspectionData>[] _currentResults;

        public void Initialize(PictureBox[] boxes)
        {
            _previewBoxes = boxes;

            for (int i = 0; i < _previewBoxes.Length; i++)
            {
                int index = i; // Closure capture

                // 設定基礎屬性
                _previewBoxes[i].Cursor = Cursors.Hand;
                _previewBoxes[i].BorderStyle = BorderStyle.None; // 接管邊框繪製

                // 綁定事件
                _previewBoxes[i].Click += (s, e) => Select(index, triggerEvent: true);
                _previewBoxes[i].Paint += OnPreviewPaint;
            }
        }

        /// <summary>
        /// 更新所有縮圖內容
        /// </summary>
        /// <param name="results">檢測結果陣列</param>
        /// <param name="cacheCollector">用於收集 Bitmap 以便 Form 進行統一 Dispose 管理的 List</param>
        public void UpdateImages(TimedResult<InspectionData>[] results, List<Image> cacheCollector)
        {
            _currentResults = results; // 保存引用

            for (int i = 0; i < _previewBoxes.Length; i++)
            {
                if (i >= results.Length) break;

                var result = results[i];
                // 取出 Image 顯示
                if (result?.Data?.Image != null)
                {
                    cacheCollector.Add(result.Data.Image);
                    _previewBoxes[i].Image = result.Data.Image;
                }
                else
                {
                    _previewBoxes[i].Image = null;
                }
                _previewBoxes[i].Invalidate();
            }
        }

        public InspectionData GetCurrentSelectionData()
        {
            if (_currentResults == null || _selectedIndex < 0 || _selectedIndex >= _currentResults.Length)
                return null;

            return _currentResults[_selectedIndex]?.Data;
        }

        /// <summary>
        /// 設定選取並更新 UI
        /// </summary>
        public void Select(int index, bool triggerEvent = true)
        {
            if (index < 0 || index >= _previewBoxes.Length) index = 0;

            _selectedIndex = index;

            // 更新所有格子的視覺狀態
            for (int i = 0; i < _previewBoxes.Length; i++)
            {
                // 使用 Tag 標記狀態，讓 Paint 事件讀取
                _previewBoxes[i].Tag = (i == index) ? "Selected" : null;
                _previewBoxes[i].Invalidate();
            }

            // 通知外部 (Form) 載入大圖
            if (triggerEvent)
            {
                SelectionChanged?.Invoke(_selectedIndex);
            }
        }

        // === 繪圖邏輯 (從 Form 移過來的) ===
        private void OnPreviewPaint(object sender, PaintEventArgs e)
        {
            if (sender is PictureBox pb)
            {
                bool isSelected = (string)pb.Tag == "Selected";

                Color borderColor = isSelected ? Color.Orange : Color.DarkGray;
                int borderWidth = isSelected ? 3 : 1;

                ControlPaint.DrawBorder(e.Graphics, pb.ClientRectangle,
                    borderColor, borderWidth, ButtonBorderStyle.Solid,
                    borderColor, borderWidth, ButtonBorderStyle.Solid,
                    borderColor, borderWidth, ButtonBorderStyle.Solid,
                    borderColor, borderWidth, ButtonBorderStyle.Solid);
            }
        }
    }
}