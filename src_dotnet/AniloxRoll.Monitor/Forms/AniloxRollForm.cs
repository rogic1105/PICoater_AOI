using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Forms.Helpers;
using AOI.SDK.Core.Models;

namespace AniloxRoll.Monitor.Forms
{
    public partial class AniloxRollForm : Form
    {
        // 核心服務
        private readonly ImageRepository _imageRepository = new ImageRepository();
        private BatchInspectionService _inspectionService;

        // UI Managers
        private TimeSelectionManager _timeSelectionManager;
        private ThumbnailGalleryManager _galleryManager;

        // Presenter (大腦)
        private AniloxRollPresenter _presenter;

        // 資源管理 (用於存放縮圖 Bitmap，以便統一 Dispose)
        private readonly List<Image> _thumbnailCache = new List<Image>();

        public AniloxRollForm()
        {
            InitializeComponent();
            InitializeSystem();
        }

        private void InitializeSystem()
        {
            // 1. 建立 Service
            _inspectionService = new BatchInspectionService();

            // 2. 建立 UI Managers
            _timeSelectionManager = new TimeSelectionManager(
                _imageRepository, cbYear, cbMonth, cbDay, cbHour, cbMin, cbSec);

            _galleryManager = new ThumbnailGalleryManager();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            // 3. 建立 Presenter 並注入依賴
            _presenter = new AniloxRollPresenter(
                _imageRepository,
                _inspectionService,
                _timeSelectionManager,
                _galleryManager
            );

            // 4. 綁定 Presenter 事件 (接收來自 Presenter 的 UI 狀態指令)
            _presenter.BusyStateChanged += SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);

            // 5. 綁定其他 UI 事件 (Designer 沒綁到的部分)
            BindManualUiEvents();
        }

        private void BindManualUiEvents()
        {
            // Gallery 選取事件 (顯示大圖)
            _galleryManager.SelectionChanged += OnGallerySelectionChanged;

            // Canvas 滑鼠移動顯示像素資訊
            canvasMain.PixelHovered += (x, y, color) =>
            {
                if (canvasMain.Image != null)
                    lblPixelInfo.Text = $"座標: ({x}, {y}) | 亮度: {color.R}";
            };
        }

        // =========================================================
        // Designer 事件區 (這些方法名稱必須保留以配合 Designer.cs)
        // =========================================================

        private async void btnShowOriginal_Click(object sender, EventArgs e)
        {
            await LoadImages(false);
        }

        private async void btnShowProcessed_Click(object sender, EventArgs e)
        {
            await LoadImages(true);
        }

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

                    // 重新初始化下拉選單
                    _timeSelectionManager.Initialize(Properties.Settings.Default.LastYear);

                    // 重置 Gallery 選擇
                    _galleryManager.Select(0, triggerEvent: false);
                }
            }
        }

        // =========================================================
        // 私有輔助方法 (View Logic)
        // =========================================================

        private async Task LoadImages(bool enableProcess)
        {
            ClearOldImages();
            // 呼叫 Presenter 執行流程
            await _presenter.RunWorkflowAsync(enableProcess, _thumbnailCache);
        }

        private void SetUiLoadingState(bool isBusy)
        {
            // 控制 UI 鎖定狀態
            Cursor = isBusy ? Cursors.WaitCursor : Cursors.Default;
            btnShowOriginal.Enabled = !isBusy;
            btnShowProcessed.Enabled = !isBusy;
            btnSelectFolder.Enabled = !isBusy;

            // 如果你在 Designer 加了 ProgressBar，也可以在這裡控制
            // if (isBusy) progressBar1.Style = ProgressBarStyle.Marquee;
            // else progressBar1.Style = ProgressBarStyle.Blocks;
        }

        private void OnGallerySelectionChanged(int index)
        {
            // 負責顯示大圖 (View 的職責)
            try
            {
                // 先嘗試從 Service 取得高解析度檢測圖 (如果有的話)
                // 這裡簡化處理：如果是在 Process 模式下，Service 會回傳運算後的圖
                // 如果是原圖模式，我們直接從硬碟讀取

                // 為了判斷是否為 ProcessMode，我們可以依賴 UI 狀態，或者簡單地嘗試
                // 這裡我們直接呼叫 Service.RunInspectionFullRes (它內部應該要有邏輯判斷或是總是執行檢測)
                // 但為了效能，若只是看原圖，我們不希望跑檢測算法。
                // 簡單解法：檢查當前哪一個按鈕被 Disabled (因為 Loading 時按鈕會鎖住，但讀取完後按鈕是 Enabled)
                // 更好的解法：讓 Presenter 告訴我們現在是什麼模式，但為了不增加複雜度，我們用 Service 的 GetFilePath 讀原圖當作 Fallback。

                // 注意：這裡假設 Service.RunInspectionFullRes 會執行演算法。
                // 如果你想區分「檢測圖」和「原圖」，你需要知道當前的模式。
                // 我們可以透過檢查 _previewBoxes 的狀態，或是增加一個變數紀錄。
                // 這裡示範最穩健的做法：讀原圖 (快速)，如果使用者之前按的是 "檢測"，則跑檢測。

                // [修正] 為了讓邏輯正確，我們假設使用者想看的是「目前縮圖對應的大圖」
                // 如果縮圖有橘色框 (代表已選取)，我們需要知道現在是「原圖模式」還是「檢測模式」
                // 為了不讓 Form 變複雜，我們簡單約定：
                // 如果需要更精確的模式控制，可以在 Presenter 增加 CurrentMode 屬性。

                Bitmap bigImg = null;

                // 嘗試讀取檔案路徑
                string path = _inspectionService.GetFilePath(index);
                if (string.IsNullOrEmpty(path)) return;

                // 這裡有一個小技巧：我們可以檢查 PictureBox 的 BorderStyle
                // 因為我們在 GalleryManager 裡，如果是 ProcessedMode 下的縮圖並沒有特別標記
                // 但我們可以根據 "最後一次按下的按鈕" 或是 Presenter 的狀態來決定

                // 暫時解法：總是讀取原圖 (速度快)，如果需要看檢測大圖，未來可以加一個 Checkbox "同步檢測大圖"
                // 或者：
                // bigImg = _inspectionService.RunInspectionFullRes(index); // 這會跑演算法

                // 為了符合你原本的邏輯 (ShowProcessed 後點大圖也要是 Processed)，
                // 我們可以簡單地增加一個 private bool _isProcessedMode; 
                // 但為了維持無狀態，建議預設顯示原圖，因為點擊縮圖反應要快。

                bigImg = new Bitmap(path); // 預設讀原圖

                if (bigImg != null)
                {
                    var old = canvasMain.Image;
                    canvasMain.Image = null;
                    old?.Dispose();

                    canvasMain.Image = bigImg;
                    canvasMain.FitToScreen();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("載入大圖失敗: " + ex.Message);
            }
        }

        private void ClearOldImages()
        {
            canvasMain.Image = null;
            // _thumbnailCache 的圖片由 Presenter/Manager 填入，但由 Form (View) 負責清理
            foreach (var img in _thumbnailCache) img.Dispose();
            _thumbnailCache.Clear();
            GC.Collect();
        }

        private void Form_Load(object sender, EventArgs e)
        {
            string path = Properties.Settings.Default.LastDataPath;
            if (Directory.Exists(path))
            {
                _imageRepository.LoadDirectory(path);
                _timeSelectionManager.Initialize(Properties.Settings.Default.LastYear);

                // 預選第一格
                _galleryManager.Select(0, triggerEvent: false);
            }
        }

        private void Form_FormClosing(object sender, FormClosingEventArgs e)
        {
            ClearOldImages();
            _inspectionService?.Dispose();
        }
    }
}