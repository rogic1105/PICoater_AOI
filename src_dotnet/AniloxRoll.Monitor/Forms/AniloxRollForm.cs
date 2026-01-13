using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;

using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AOI.SDK.Core.Models; // 引用外部 SDK

namespace AniloxRoll.Monitor.Forms
{
    public partial class AniloxRollForm : Form
    {
        // 狀態與設定
        private int _fullImageWidth, _fullImageHeight;
        private bool _isProcessedMode = false;
        private bool _isLoading = false;

        // 核心組件
        private ImageRepository _imageRepository = new ImageRepository();
        private PICoaterProcessor[] _processors = new PICoaterProcessor[7];

        // UI 參考與快取
        private PictureBox[] _previewBoxes;
        private string[] _currentFilePaths = new string[7];
        private List<Image> _thumbnailCache = new List<Image>();

        public AniloxRollForm()
        {
            InitializeComponent();
            InitializeControls();
            InitializeProcessors();
            SetupCanvasEvents();
        }

        private void InitializeControls()
        {
            // 綁定 UI 上的 PictureBox
            _previewBoxes = new PictureBox[] { pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7 };

            for (int i = 0; i < 7; i++)
            {
                int index = i;
                _previewBoxes[i].Click += (s, e) => OnPreviewClick(index);
                _previewBoxes[i].Cursor = Cursors.Hand;
            }
        }

        private void InitializeProcessors()
        {
            // 每個相機對應一個處理器
            for (int i = 0; i < 7; i++)
            {
                _processors[i] = new PICoaterProcessor();
            }
        }

        private void SetupCanvasEvents()
        {
            // 假設 canvasMain 是你的 SmartCanvas 控件
            canvasMain.PixelHovered += (x, y, color) =>
            {
                if (x >= 0 && x < _fullImageWidth && y >= 0 && y < _fullImageHeight)
                {
                    lblPixelInfo.Text = $"座標: ({x}, {y}) | 亮度: {color.R}";
                }
            };
        }

        private void Form_Load(object sender, EventArgs e)
        {
            string path = Properties.Settings.Default.LastDataPath;
            if (Directory.Exists(path))
            {
                _imageRepository.LoadDirectory(path);
                UpdateCombo(cbYear, _imageRepository.GetYears(), Properties.Settings.Default.LastYear);

                // 自動選取第一項
                if (cbYear.Items.Count > 0 && cbYear.SelectedIndex == -1)
                    cbYear.SelectedIndex = 0;
            }
        }

        private void Form_FormClosing(object sender, FormClosingEventArgs e)
        {
            // 清理圖片快取
            foreach (var img in _thumbnailCache) img?.Dispose();
            if (canvasMain.Image != null) canvasMain.Image.Dispose();

            // 清理處理器資源
            for (int i = 0; i < 7; i++) _processors[i]?.Dispose();
        }

        // === 按鈕功能: 選擇資料夾 ===
        private void btnSelectFolder_Click(object sender, EventArgs e)
        {
            using (var fbd = new FolderBrowserDialog())
            {
                if (Directory.Exists(Properties.Settings.Default.LastDataPath))
                    fbd.SelectedPath = Properties.Settings.Default.LastDataPath;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    _imageRepository.LoadDirectory(fbd.SelectedPath);

                    if (_imageRepository.FileCount == 0)
                    {
                        MessageBox.Show("該路徑下無符合格式的圖片！");
                        return;
                    }

                    Properties.Settings.Default.LastDataPath = fbd.SelectedPath;
                    Properties.Settings.Default.Save();

                    UpdateCombo(cbYear, _imageRepository.GetYears(), Properties.Settings.Default.LastYear);
                    if (cbYear.Items.Count > 0 && cbYear.SelectedIndex == -1)
                        cbYear.SelectedIndex = 0;
                }
            }
        }

        // === 下拉選單連動 ===
        private void cbYear_SelectedIndexChanged(object sender, EventArgs e) =>
            UpdateCombo(cbMonth, _imageRepository.GetMonths(cbYear.Text), Properties.Settings.Default.LastMonth);

        private void cbMonth_SelectedIndexChanged(object sender, EventArgs e) =>
            UpdateCombo(cbDay, _imageRepository.GetDays(cbYear.Text, cbMonth.Text), Properties.Settings.Default.LastDay);

        private void cbDay_SelectedIndexChanged(object sender, EventArgs e) =>
            UpdateCombo(cbHour, _imageRepository.GetHours(cbYear.Text, cbMonth.Text, cbDay.Text), Properties.Settings.Default.LastHour);

        private void cbHour_SelectedIndexChanged(object sender, EventArgs e) =>
            UpdateCombo(cbMin, _imageRepository.GetMinutes(cbYear.Text, cbMonth.Text, cbDay.Text, cbHour.Text), Properties.Settings.Default.LastMin);

        private void cbMin_SelectedIndexChanged(object sender, EventArgs e) =>
            UpdateCombo(cbSec, _imageRepository.GetSeconds(cbYear.Text, cbMonth.Text, cbDay.Text, cbHour.Text, cbMin.Text), Properties.Settings.Default.LastSec);

        private void cbSec_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbSec.SelectedItem != null) SaveCurrentSelection();
        }

        private void UpdateCombo(ComboBox cb, List<string> items, string lastVal)
        {
            cb.Items.Clear();
            cb.Items.AddRange(items.ToArray());
            if (items.Count == 0) return;
            cb.SelectedItem = items.Contains(lastVal) ? lastVal : items[0];
        }

        private void SaveCurrentSelection()
        {
            Properties.Settings.Default.LastYear = cbYear.Text;
            Properties.Settings.Default.LastMonth = cbMonth.Text;
            Properties.Settings.Default.LastDay = cbDay.Text;
            Properties.Settings.Default.LastHour = cbHour.Text;
            Properties.Settings.Default.LastMin = cbMin.Text;
            Properties.Settings.Default.LastSec = cbSec.Text;
            Properties.Settings.Default.Save();
        }

        // === 按鈕功能: 顯示原圖 / 顯示檢測圖 ===
        private async void btnShowOriginal_Click(object sender, EventArgs e) =>
            await LoadImagesAsync(enableProcess: false);

        private async void btnShowProcessed_Click(object sender, EventArgs e) =>
            await LoadImagesAsync(enableProcess: true);

        private async Task LoadImagesAsync(bool enableProcess)
        {
            if (_isLoading) return;
            _isLoading = true;
            _isProcessedMode = enableProcess;

            // UI 鎖定
            Cursor = Cursors.WaitCursor;
            btnShowOriginal.Enabled = false;
            btnShowProcessed.Enabled = false;

            // 1. 清除舊圖 & 資源釋放
            canvasMain.Image = null;
            foreach (var img in _thumbnailCache) img.Dispose();
            _thumbnailCache.Clear();
            GC.Collect();

            // 取得檔案路徑
            var filesMap = _imageRepository.GetImages(
                cbYear.Text, cbMonth.Text, cbDay.Text,
                cbHour.Text, cbMin.Text, cbSec.Text);

            // 準備暫存陣列 (用來存放 7 張圖的結果，確保順序)
            var tempResults = new TimedResult<Bitmap>[7];
            var logs = new ConcurrentQueue<string>();

            var swTotal = Stopwatch.StartNew();

            try
            {
                await Task.Run(() =>
                {
                    // 限制平行數量為 4，避免 SSD 讀取塞車 (保持速度優勢)
                    var pOptions = new ParallelOptions { MaxDegreeOfParallelism = 4 };

                    Parallel.For(0, 7, pOptions, i =>
                    {
                        int camId = i + 1;
                        string path = filesMap.ContainsKey(camId) ? filesMap[camId] : null;
                        _currentFilePaths[i] = path;

                        if (!string.IsNullOrEmpty(path))
                        {
                            // 執行運算或讀取
                            if (enableProcess)
                            {
                                tempResults[i] = _processors[i].ProcessImage(path, 1000);
                            }
                            else
                            {
                                tempResults[i] = _processors[i].LoadThumbnailOnly(path, 1000);
                            }

                            // 記錄 Log (但不更新 UI)
                            if (tempResults[i] != null)
                            {
                                logs.Enqueue($"Cam {camId}: IO={tempResults[i].IoDurationMs}ms, GPU={tempResults[i].ComputeDurationMs}ms");
                            }
                        }
                        else
                        {
                            tempResults[i] = null;
                        }
                    });
                });

                // === [關鍵修改] 等上面全部跑完，回到 UI Thread 一次性更新 ===
                // 這樣 7 張圖會「同時」出現，不會有頓挫感

                for (int i = 0; i < 7; i++)
                {
                    var result = tempResults[i];
                    if (result?.Data != null)
                    {
                        // 加入 Cache 管理生命週期
                        _thumbnailCache.Add(result.Data);

                        // 設定圖片
                        _previewBoxes[i].Image = result.Data;
                        _previewBoxes[i].BorderStyle = enableProcess ? BorderStyle.FixedSingle : BorderStyle.None;
                    }
                    else
                    {
                        _previewBoxes[i].Image = null;
                        _previewBoxes[i].BorderStyle = BorderStyle.None;
                    }
                }
            }
            finally
            {
                _isLoading = false;
                Cursor = Cursors.Default;
                btnShowOriginal.Enabled = true;
                btnShowProcessed.Enabled = true;

                swTotal.Stop();
                ShowPerformanceReport(swTotal.ElapsedMilliseconds, logs);
            }
        }

        private void ShowPerformanceReport(long totalMs, ConcurrentQueue<string> logs)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"總耗時: {totalMs} ms");
            sb.AppendLine("--------------------------------");

            var sortedLogs = new List<string>(logs);
            sortedLogs.Sort();
            foreach (var log in sortedLogs) sb.AppendLine(log);

            Console.WriteLine(sb.ToString()); // 可改為顯示在 Label
        }

        // === 點擊縮圖：顯示大圖 ===
        private void OnPreviewClick(int index)
        {
            string path = _currentFilePaths[index];
            if (string.IsNullOrEmpty(path)) return;

            try
            {
                if (canvasMain.Image != null)
                {
                    var old = canvasMain.Image;
                    canvasMain.Image = null;
                    old.Dispose();
                    GC.Collect();
                }

                Bitmap bigImg = null;

                if (_isProcessedMode)
                {
                    // 檢測模式：跑完整檢測演算法
                    bigImg = _processors[index].RunInspectionFullRes(path);
                }
                else
                {
                    // 原圖模式：直接讀取
                    bigImg = new Bitmap(path);
                }

                if (bigImg != null)
                {
                    _fullImageWidth = bigImg.Width;
                    _fullImageHeight = bigImg.Height;
                    canvasMain.Image = bigImg;
                    canvasMain.FitToScreen();
                }

                // Highlight 選取框
                foreach (var pb in _previewBoxes) pb.BackColor = Color.Transparent;
                _previewBoxes[index].BackColor = Color.Orange;
            }
            catch (Exception ex)
            {
                MessageBox.Show("載入大圖失敗: " + ex.Message);
            }
        }




    }
}