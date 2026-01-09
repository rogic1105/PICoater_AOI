using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging; // 用於 PixelFormat
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using AOI.SDK.UI; // 你的 SmartCanvas namespace

namespace AniloxRoll.Monitor
{
    public partial class Form1 : Form
    {
        // === 1. 成員變數 ===
        private ImageDataManager _dataMgr = new ImageDataManager();

        // 7 個預覽框 (對應界面上的 pbCam1 ~ pbCam7)
        private PictureBox[] _previewBoxes = new PictureBox[7];

        // 當前 7 張圖的檔案路徑
        private string[] _currentFilePaths = new string[7];

        // 縮圖快取 (用來在 Form 關閉時釋放)
        private List<Image> _thumbnailCache = new List<Image>();

        // 演算法 Wrapper (隨 Form 建立，程式結束時 Dispose)
        private PICoaterWrapper _algo = new PICoaterWrapper();

        // 防呆旗標：防止使用者在讀圖時重複點擊
        private bool _isLoading = false;

        public Form1()
        {
            InitializeComponent();
            InitializeControls();
        }

        private void InitializeControls()
        {
            // 將 Designer 上的 PictureBox 放入陣列
            _previewBoxes[0] = pbCam1;
            _previewBoxes[1] = pbCam2;
            _previewBoxes[2] = pbCam3;
            _previewBoxes[3] = pbCam4;
            _previewBoxes[4] = pbCam5;
            _previewBoxes[5] = pbCam6;
            _previewBoxes[6] = pbCam7;

            // 綁定點擊事件：點縮圖 -> 顯示大圖
            for (int i = 0; i < 7; i++)
            {
                int index = i; // 閉包變數
                _previewBoxes[i].Click += (s, e) => OnPreviewClick(index);
                _previewBoxes[i].Cursor = Cursors.Hand;
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // 讀取上次的路徑設定
            string path = Properties.Settings.Default.LastDataPath;
            if (Directory.Exists(path))
            {
                _dataMgr.LoadDirectory(path);
                UpdateCombo(cbYear, _dataMgr.GetYears(), Properties.Settings.Default.LastYear);

                // 自動觸發連動 (如果只有一個選項)
                if (cbYear.Items.Count > 0 && cbYear.SelectedIndex == -1) cbYear.SelectedIndex = 0;
            }
        }

        // === 2. 選單與路徑邏輯 ===

        private void btnSelectFolder_Click(object sender, EventArgs e)
        {
            using (var fbd = new FolderBrowserDialog())
            {
                if (Directory.Exists(Properties.Settings.Default.LastDataPath))
                {
                    fbd.SelectedPath = Properties.Settings.Default.LastDataPath;
                }

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

                    // 自動觸發連動
                    if (cbYear.Items.Count > 0 && cbYear.SelectedIndex == -1) cbYear.SelectedIndex = 0;
                }
            }
        }

        // --- Cascading Dropdowns ---
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
            else cb.SelectedIndex = 0; // 自動選第一個
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

        // === 3. 核心功能：載入圖片 ===

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
            if (_isLoading) return; // 防呆：避免重複執行

            try
            {
                _isLoading = true;
                this.Cursor = Cursors.WaitCursor;
                btnShowOriginal.Enabled = false;
                btnShowProcessed.Enabled = false;

                // 1. 取得路徑
                var filesMap = _dataMgr.GetImages(cbYear.Text, cbMonth.Text, cbDay.Text, cbHour.Text, cbMin.Text, cbSec.Text);

                // 2. 清理資源
                mainCanvas.Image = null;
                foreach (var img in _thumbnailCache) img.Dispose();
                _thumbnailCache.Clear();
                GC.Collect(); // 強制回收大圖

                // 3. 執行載入
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

                                if (enableProcess)
                                {
                                    // A. 檢測模式 (需讀大圖 -> 跑 DLL -> 縮圖)
                                    // 注意：GDI+ 讀大圖較慢
                                    using (Bitmap original = new Bitmap(path))
                                    {
                                        // 關鍵：Wrapper 內部使用同一個 GPU Context，必須鎖定避免多執行緒衝突
                                        Bitmap result;
                                        lock (_algo)
                                        {
                                            result = _algo.RunInspection(original);
                                        }

                                        using (result) // 用完要 Dispose
                                        {
                                            Bitmap thumb = MakeThumbnail(result, 1000);
                                            lock (_thumbnailCache) { _thumbnailCache.Add(thumb); }
                                            this.Invoke((Action)(() => _previewBoxes[i].Image = thumb));
                                        }
                                    }
                                }
                                else
                                {
                                    // B. 原圖模式 (C++ 極速縮圖)
                                    Bitmap thumb = PICoaterWrapper.LoadThumbnail(path, 1000);
                                    if (thumb != null)
                                    {
                                        lock (_thumbnailCache) { _thumbnailCache.Add(thumb); }
                                        this.Invoke((Action)(() => _previewBoxes[i].Image = thumb));
                                    }
                                }
                            }
                        }
                    });
                });
            }
            finally
            {
                _isLoading = false;
                this.Cursor = Cursors.Default;
                btnShowOriginal.Enabled = true;
                btnShowProcessed.Enabled = true;
            }
        }

        private Bitmap MakeThumbnail(Bitmap src, int width)
        {
            int height = (int)((float)src.Height / src.Width * width);
            Bitmap thumb = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(thumb))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(src, 0, 0, width, height);
            }
            return thumb;
        }

        private void OnPreviewClick(int index)
        {
            string path = _currentFilePaths[index];
            if (string.IsNullOrEmpty(path)) return;

            try
            {
                if (mainCanvas.Image != null)
                {
                    var old = mainCanvas.Image;
                    mainCanvas.Image = null;
                    old.Dispose();
                    GC.Collect();
                }

                Bitmap bigImg = new Bitmap(path);

                // 如果是在 "檢測模式" 下，這裡其實應該顯示檢測後的大圖
                // 但為了簡單，目前點擊預覽還是先顯示原圖
                // 若要顯示檢測圖，需再次呼叫 _algo.RunInspection(bigImg) 並顯示結果

                mainCanvas.Image = bigImg;
                mainCanvas.FitToScreen();

                foreach (var pb in _previewBoxes) pb.BorderStyle = BorderStyle.None;
                _previewBoxes[index].BorderStyle = BorderStyle.Fixed3D;
            }
            catch (Exception ex)
            {
                MessageBox.Show("載入大圖失敗: " + ex.Message);
            }
        }

        // 關閉視窗時清理所有資源
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            foreach (var img in _thumbnailCache) img?.Dispose();
            if (mainCanvas.Image != null) mainCanvas.Image.Dispose();

            // 釋放 DLL 資源
            _algo.Dispose();
        }
    }
}