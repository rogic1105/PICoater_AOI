// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Forms\MainForm.cs

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;

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
        private PICoaterWrapper _algo = new PICoaterWrapper();

        private bool _isLoading = false;

        public MainForm()
        {
            InitializeComponent();
            InitializeControls();

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

            try
            {
                // 1. 狀態設定
                _isLoading = true;
                _isProcessedMode = enableProcess; // 記錄當前模式
                this.Cursor = Cursors.WaitCursor;
                btnShowOriginal.Enabled = false;
                btnShowProcessed.Enabled = false;

                // 取得檔案路徑
                var filesMap = _dataMgr.GetImages(
                    cbYear.Text, cbMonth.Text, cbDay.Text,
                    cbHour.Text, cbMin.Text, cbSec.Text);

                // 2. 清理資源
                canvasMain.Image = null;
                foreach (var img in _thumbnailCache) img.Dispose();
                _thumbnailCache.Clear();
                GC.Collect(); // 強制回收記憶體

                // 3. 非同步並行處理 (7 張圖同時跑)
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
                                Bitmap thumb = null;

                                if (enableProcess)
                                {
                                    // === [重點] 檢測模式 ===
                                    // 流程：FastRead -> C++ Run -> 取得 Result Big -> 縮圖 -> Dispose Big

                                    // 為了線程安全 (C++ Context 共用)，這裡加鎖
                                    // 如果你的 C++ DLL 內部已經支援多執行緒 Context (每個 PICoaterWrapper 獨立)，可以不用鎖
                                    // 假設目前 _algo 是共用的，建議鎖住避免 CUDA Stream 衝突
                                    Bitmap resultBig = null;
                                    lock (_algo)
                                    {
                                        // 這行會呼叫 C++ 的 PICoater_Run (即 PICoaterDetector::Run)
                                        resultBig = _algo.RunInspectionFast(path);
                                    }

                                    if (resultBig != null)
                                    {
                                        using (resultBig) // 用完馬上釋放 100MB 大圖，只留縮圖
                                        {
                                            thumb = BitmapHelper.MakeThumbnail(resultBig, 1000);
                                        }
                                    }
                                }
                                else
                                {
                                    // === 原圖模式 ===
                                    // 使用 C++ 極速縮圖讀取 (LoadThumbnail)
                                    thumb = PICoaterWrapper.LoadThumbnail(path, 1000);
                                }

                                // 更新 UI (縮圖)
                                if (thumb != null)
                                {
                                    lock (_thumbnailCache) { _thumbnailCache.Add(thumb); }
                                    this.Invoke((Action)(() =>
                                    {
                                        _previewBoxes[i].Image = thumb;
                                        // 可以在這裡加個邊框顏色區分，例如紅色代表檢測過
                                        if (enableProcess) _previewBoxes[i].BorderStyle = BorderStyle.FixedSingle;
                                        else _previewBoxes[i].BorderStyle = BorderStyle.None;
                                    }));
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
            }
        }

        private void OnPreviewClick(int index)
        {
            string path = _currentFilePaths[index];
            if (string.IsNullOrEmpty(path)) return;

            try
            {
                // 清理舊的大圖
                if (canvasMain.Image != null)
                {
                    var old = canvasMain.Image;
                    canvasMain.Image = null;
                    old.Dispose();
                    GC.Collect();
                }

                Bitmap bigImg = null;

                // [關鍵] 根據當前模式，決定大圖要不要運算
                if (_isProcessedMode)
                {
                    // 如果現在是檢測模式，點擊縮圖時，應該要重新算一次大圖給 Canvas 看
                    // (因為之前算完的 resultBig 已經 Dispose 了，這樣才省記憶體)
                    lock (_algo)
                    {
                        bigImg = _algo.RunInspectionFast(path);
                    }
                }
                else
                {
                    // 原圖模式，直接讀檔
                    // 也可以用 FastRead 加速：
                    // bigImg = LoadUsingFastRead(path); // 若有實作
                    // 目前先用 GDI+
                    bigImg = new Bitmap(path);
                }

                if (bigImg != null)
                {
                    _imgW = bigImg.Width;
                    _imgH = bigImg.Height;
                    canvasMain.Image = bigImg;
                    canvasMain.FitToScreen();
                }

                // 更新選取狀態框
                foreach (var pb in _previewBoxes)
                    pb.BackColor = Color.Transparent; // 或控制 Padding/Border

                _previewBoxes[index].BackColor = Color.Orange; // 簡單示意選取
            }
            catch (Exception ex)
            {
                MessageBox.Show("載入大圖失敗: " + ex.Message);
            }
        }


        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            foreach (var img in _thumbnailCache) img?.Dispose();
            if (canvasMain.Image != null) canvasMain.Image.Dispose();
            _algo.Dispose();
        }
    }
}