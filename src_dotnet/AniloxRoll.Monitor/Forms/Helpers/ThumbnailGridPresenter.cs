using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using AOI.SDK.Core.Models;

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
        public void UpdateImages(TimedResult<Bitmap>[] results, List<Image> cacheCollector)
        {
            for (int i = 0; i < _previewBoxes.Length; i++)
            {
                if (i >= results.Length) break;

                var result = results[i];
                if (result?.Data != null)
                {
                    // 加入外部 Cache List 管理生命週期
                    cacheCollector.Add(result.Data);
                    _previewBoxes[i].Image = result.Data;
                }
                else
                {
                    _previewBoxes[i].Image = null;
                }

                // 強制重繪以確保邊框在最上層
                _previewBoxes[i].Invalidate();
            }
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