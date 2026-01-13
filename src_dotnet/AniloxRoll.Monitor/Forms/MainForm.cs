// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Forms\MainForm.cs

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections.Concurrent; // [新增] 用於收集多執行緒Log
using System.Diagnostics;          // [新增] 用於計時
using System.Text;                 // [新增] 用於組字串

using AniloxRoll.Monitor.Core;
using AniloxRoll.Monitor.Utils;

namespace AniloxRoll.Monitor.Forms
{
    public partial class MainForm : Form
    {
        private int _imgW, _imgH;
        private bool _isProcessedMode = false;

        // 使用 Core 中的 Manager
        private ImageDataManager _dataMgr = new ImageDataManager();

        private PictureBox[] _previewBoxes = new PictureBox[7];
        private string[] _currentFilePaths = new string[7];
        private List<Image> _thumbnailCache = new List<Image>();

        // 使用 Core 中的 Wrapper
        private PICoaterWrapper[] _algoPool = new PICoaterWrapper[7];

        private bool _isLoading = false;

        public MainForm()
        {
            InitializeComponent();
            InitializeControls();
            InitializeAlgoPool();

            // 使用 AOI.SDK.UI.SmartCanvas 的事件
            canvasMain.PixelHovered += (x, y, color) =>
            {
                // 注意：_imgW, _imgH 需在讀圖時更新
                if (x >= 0 && x < _imgW && y >= 0 && y < _imgH)
                {
                    lblPixelInfo.Text = $"座標: ({x}, {y}) | 亮度: {color.R}";
                }
            };
        }

        private void InitializeControls()
        {
            _previewBoxes[0] = pbCam1;
            _previewBoxes[1] = pbCam2;
            _previewBoxes[2] = pbCam3;
            _previewBoxes[3] = pbCam4;
            _previewBoxes[4] = pbCam5;
            _previewBoxes[5] = pbCam6;
            _previewBoxes[6] = pbCam7;

            for (int i = 0; i < 7; i++)
            {
                int index = i;
                _previewBoxes[i].Click += (s, e) => OnPreviewClick(index);
                _previewBoxes[i].Cursor = Cursors.Hand;
            }
        }

        private void InitializeAlgoPool()
        {
            for (int i = 0; i < 7; i++)
            {
                _algoPool[i] = new PICoaterWrapper();
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            string path = Properties.Settings.Default.LastDataPath;
            if (Directory.Exists(path))
            {
                _dataMgr.LoadDirectory(path);
                UpdateCombo(cbYear, _dataMgr.GetYears(), Properties.Settings.Default.LastYear);
                if (cbYear.Items.Count > 0 && cbYear.SelectedIndex == -1) cbYear.SelectedIndex = 0;
            }
        }

        // === 下拉選單邏輯 (保持不變) ===
        private void btnSelectFolder_Click(object sender, EventArgs e)
        {
            using (var fbd = new FolderBrowserDialog())
            {
                if (Directory.Exists(Properties.Settings.Default.LastDataPath))
                    fbd.SelectedPath = Properties.Settings.Default.LastDataPath;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    string path = fbd.SelectedPath;
                    _dataMgr.LoadDirectory(path);

                    if (_dataMgr.FileCount == 0)
                    {
                        MessageBox.Show("該路徑下無符合格式的圖片！");
                        return;
                    }

                    Properties.Settings.Default.LastDataPath = path;
                    Properties.Settings.Default.Save();

                    UpdateCombo(cbYear, _dataMgr.GetYears(), Properties.Settings.Default.LastYear);
                    if (cbYear.Items.Count > 0 && cbYear.SelectedIndex == -1) cbYear.SelectedIndex = 0;
                }
            }
        }

        private void cbYear_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbYear.SelectedItem == null) return;
            UpdateCombo(cbMonth, _dataMgr.GetMonths(cbYear.Text), Properties.Settings.Default.LastMonth);
        }
        private void cbMonth_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbMonth.SelectedItem == null) return;
            UpdateCombo(cbDay, _dataMgr.GetDays(cbYear.Text, cbMonth.Text), Properties.Settings.Default.LastDay);
        }
        private void cbDay_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbDay.SelectedItem == null) return;
            UpdateCombo(cbHour, _dataMgr.GetHours(cbYear.Text, cbMonth.Text, cbDay.Text), Properties.Settings.Default.LastHour);
        }
        private void cbHour_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbHour.SelectedItem == null) return;
            UpdateCombo(cbMin, _dataMgr.GetMinutes(cbYear.Text, cbMonth.Text, cbDay.Text, cbHour.Text), Properties.Settings.Default.LastMin);
        }
        private void cbMin_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbMin.SelectedItem == null) return;
            UpdateCombo(cbSec, _dataMgr.GetSeconds(cbYear.Text, cbMonth.Text, cbDay.Text, cbHour.Text, cbMin.Text), Properties.Settings.Default.LastSec);
        }
        private void cbSec_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (cbSec.SelectedItem != null) SaveCurrentSelection();
        }

        private void UpdateCombo(ComboBox cb, List<string> items, string lastVal)
        {
            cb.Items.Clear();
            cb.Items.AddRange(items.ToArray());
            if (items.Count == 0) return;
            if (items.Contains(lastVal)) cb.SelectedItem = lastVal;
            else cb.SelectedIndex = 0;
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

        // === 載入圖片邏輯 ===

        private async void btnShowOriginal_Click(object sender, EventArgs e)
        {
            await LoadImagesAsync(enableProcess: false);
        }

        private async void btnShowProcessed_Click(object sender, EventArgs e)
        {
            await LoadImagesAsync(enableProcess: true);
        }

        private async Task LoadImagesAsync(bool enableProcess)
        {
            if (_isLoading) return;

            Stopwatch swTotal = Stopwatch.StartNew();
            ConcurrentQueue<string> debugLogs = new ConcurrentQueue<string>();

            try
            {
                _isLoading = true;
                _isProcessedMode = enableProcess;
                this.Cursor = Cursors.WaitCursor;
                btnShowOriginal.Enabled = false;
                btnShowProcessed.Enabled = false;

                var filesMap = _dataMgr.GetImages(
                    cbYear.Text, cbMonth.Text, cbDay.Text,
                    cbHour.Text, cbMin.Text, cbSec.Text);

                canvasMain.Image = null;
                foreach (var img in _thumbnailCache) img.Dispose();
                _thumbnailCache.Clear();
                GC.Collect();
                GC.WaitForPendingFinalizers();

                await Task.Run(() =>
                {
                    Parallel.For(0, 7, i =>
                    {
                        int camId = i + 1;
                        _currentFilePaths[i] = null;

                        if (filesMap.ContainsKey(camId))
                        {
                            string path = filesMap[camId];
                            if (File.Exists(path))
                            {
                                _currentFilePaths[i] = path;

                                // 變數宣告
                                Bitmap thumb = null;
                                long t_io = 0;
                                long t_gpu = 0;

                                if (enableProcess)
                                {
                                    // === 檢測模式 (詳細計時版) ===
                                    // 使用 Wrapper 直接回傳時間資訊
                                    var result = _algoPool[i].RunInspectionFast_Detailed(path, 1000);

                                    thumb = result.Thumbnail;
                                    t_io = result.IoTimeMs;
                                    t_gpu = result.GpuTimeMs;

                                    // result 本身不需要 Dispose (除了 Thumbnail)，因為它是純數據結構
                                }
                                else
                                {
                                    // === 原圖模式 ===
                                    Stopwatch sw = Stopwatch.StartNew();
                                    thumb = PICoaterWrapper.LoadThumbnail(path, 1000);
                                    sw.Stop();
                                    t_io = sw.ElapsedMilliseconds; // 原圖模式幾乎都是 IO 時間
                                }

                                if (thumb != null)
                                {
                                    lock (_thumbnailCache) { _thumbnailCache.Add(thumb); }
                                    this.Invoke((Action)(() =>
                                    {
                                        _previewBoxes[i].Image = thumb;
                                        if (enableProcess) _previewBoxes[i].BorderStyle = BorderStyle.FixedSingle;
                                        else _previewBoxes[i].BorderStyle = BorderStyle.None;
                                    }));

                                    // [記錄] 區分 IO 和 GPU 時間
                                    debugLogs.Enqueue($"Cam {camId}: IO(讀碟)={t_io}ms, GPU(運算)={t_gpu}ms");
                                }
                            }
                        }
                        else
                        {
                            this.Invoke((Action)(() => _previewBoxes[i].Image = null));
                        }
                    });
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"處理失敗: {ex.Message}");
            }
            finally
            {
                _isLoading = false;
                this.Cursor = Cursors.Default;
                btnShowOriginal.Enabled = true;
                btnShowProcessed.Enabled = true;

                swTotal.Stop();
                ShowPerformanceReport(swTotal.ElapsedMilliseconds, debugLogs);
            }
        }



        // [新增] 顯示統計結果的輔助函式
        private void ShowPerformanceReport(long totalMs, ConcurrentQueue<string> logs)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"總耗時: {totalMs} ms");
            sb.AppendLine("--------------------------------");

            // 排序讓 Cam 1~7 依序顯示
            var sortedLogs = new List<string>(logs);
            sortedLogs.Sort();

            foreach (var log in sortedLogs)
            {
                sb.AppendLine(log);
            }

            // 顯示在 MessageBox，或是你可以 output 到 Console / Label
            // MessageBox.Show(sb.ToString(), "效能分析報告"); 

            // 建議改寫到 Label 或 Console，不然每次彈窗很煩
            Console.WriteLine(sb.ToString());
            // 或者如果你介面上有個 statusLabel:
            // lblPixelInfo.Text = $"Done in {totalMs} ms";
        }

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
                    // [修改] 點擊縮圖時，也使用專屬的 Wrapper 重算
                    // 這會非常快，因為 GPU 記憶體已經分配好了，只是重新 Upload + Run
                    bigImg = _algoPool[index].RunInspectionFast(path);
                }
                else
                {
                    bigImg = new Bitmap(path);
                }

                if (bigImg != null)
                {
                    _imgW = bigImg.Width;
                    _imgH = bigImg.Height;
                    canvasMain.Image = bigImg;
                    canvasMain.FitToScreen();
                }

                foreach (var pb in _previewBoxes) pb.BackColor = Color.Transparent;
                _previewBoxes[index].BackColor = Color.Orange;
            }
            catch (Exception ex)
            {
                MessageBox.Show("載入大圖失敗: " + ex.Message);
            }
        }

        // [修改] 程式關閉時，一次釋放所有 Wrapper
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            foreach (var img in _thumbnailCache) img?.Dispose();
            if (canvasMain.Image != null) canvasMain.Image.Dispose();

            // 釋放 7 個 Wrapper (釋放 3.5GB VRAM)
            for (int i = 0; i < 7; i++)
            {
                _algoPool[i]?.Dispose();
            }
        }

    }
}