// TanukiCv.BenchUi\Forms\SdkForm.cs

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Forms;

using TanukiCv.Core;
using TanukiCv.Utils;          // ImageUtils（仍在 Core）
using TanukiCv.BenchUi.Utils;  // FileUtils（已移來 BenchUi）

namespace TanukiCv.BenchUi.Forms
{
    public partial class SdkForm : Form
    {
        // ---------------------------------------------------------
        // 變數宣告
        // ---------------------------------------------------------
        private byte[] _originalData; // 原始圖片數據 (8bpp)
        private byte[] _currentData;  // 目前顯示的圖片數據
        private int _imgW, _imgH;

        private GPUProcessor _gpu = new GPUProcessor();

        public SdkForm()
        {
            InitializeComponent();

            // 非同步執行 GPU 暖身
            _ = GPUHelper.WarmUpAsync();

            // [修改] 綁定 SmartCanvas 的 StatusChanged 事件
            // 配合 SmartCanvas 的更新，顯示完整的座標、亮度、縮放倍率與平移位置
            canvasMain.StatusChanged += (info) =>
            {
                // SmartCanvas 內部已經處理過座標邊界與有效性 (ImageX/Y 是最後一次有效位置)
                // 直接顯示 CanvasInfo 提供的資訊即可
                lblPixelInfo.Text = $"座標: ({info.ImageX}, {info.ImageY}) | " +
                                    $"亮度: {info.PixelColor.R} | " +
                                    $"倍率: {info.Zoom:F5}x | " +
                                    $"平移: ({info.PanOffset.X:F0}, {info.PanOffset.Y:F0})";
            };
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            _gpu.Dispose();
            base.OnFormClosed(e);
        }


        // ---------------------------------------------------------
        // 3. 極速讀取圖片 (Fast IO)
        // ---------------------------------------------------------
        private void btnLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "BMP Files|*.bmp|All Files|*.*";

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                Stopwatch sw = Stopwatch.StartNew();

                // 1. Host Pinned Memory Alloc (預估最大值，確保夠用)
                int maxW = 16384;
                int maxH = 10000;
                ulong maxBytes = (ulong)(maxW * maxH);

                // 使用 Wrapper 分配鎖頁記憶體
                IntPtr pBuffer = TanukiCvWrapper.TanukiCv_AllocPinned(maxBytes);

                try
                {
                    int w, h;
                    // 呼叫 C++ 極速讀檔
                    bool success = TanukiCvWrapper.TanukiCv_FastReadBMP(ofd.FileName, out w, out h, pBuffer, (int)maxBytes);

                    if (success)
                    {
                        _imgW = w;
                        _imgH = h;
                        int realSize = w * h;

                        // 準備 Host 端緩衝區
                        _originalData = new byte[realSize];
                        Marshal.Copy(pBuffer, _originalData, 0, realSize);

                        _currentData = new byte[realSize];
                        Array.Copy(_originalData, _currentData, realSize);

                        // [關鍵] 初始化 GPU Processor
                        // 這一步會自動分配 GPU 記憶體並上傳原圖
                        _gpu.Initialize(pBuffer, w, h);

                        sw.Stop();
                        lblLoadTime.Text = $"{sw.Elapsed.TotalMilliseconds:F1} ms";

                        // 更新 UI
                        txtLoadPath.Text = ofd.FileName;
                        txtSavePath.Text = ""; // 清空存檔路徑
                        ShowImage(_currentData);
                        canvasMain.FitToScreen(); // 自動縮放適配視窗
                    }
                    else
                    {
                        MessageBox.Show("讀取失敗！請確認檔案格式是否為 8-bit BMP。");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"發生錯誤: {ex.Message}");
                }
                finally
                {
                    // 務必釋放 Pinned Memory
                    TanukiCvWrapper.TanukiCv_FreePinned(pBuffer);
                }
            }
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            if (_currentData == null) { MessageBox.Show("沒有圖片可存！"); return; }

            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "BMP Files|*.bmp";
            sfd.FileName = "output_fast.bmp";

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                Stopwatch sw = Stopwatch.StartNew();

                // 鎖定 managed array 以便傳給 C++
                GCHandle hData = GCHandle.Alloc(_currentData, GCHandleType.Pinned);
                try
                {
                    IntPtr ptrData = hData.AddrOfPinnedObject();
                    // 呼叫 C++ 極速寫檔
                    bool success = TanukiCvWrapper.TanukiCv_FastWriteBMP(sfd.FileName, _imgW, _imgH, ptrData);

                    sw.Stop();
                    lblSaveTime.Text = $"{sw.Elapsed.TotalMilliseconds:F1} ms";

                    if (success)
                    {
                        txtSavePath.Text = sfd.FileName;
                    }
                    else
                    {
                        MessageBox.Show("存檔失敗！");
                    }
                }
                finally
                {
                    hData.Free();
                }
            }
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            if (_originalData == null) return;

            // Host 端還原
            Array.Copy(_originalData, _currentData, _originalData.Length);
            ShowImage(_currentData);

            // GPU 端還原 (透過 Upload 重新覆蓋 Current)
            _gpu.Reset(_originalData);
        }
       
        private void btnOpenLoadDir_Click(object sender, EventArgs e)
        {
            FileUtils.OpenFolderAndSelectFile(txtLoadPath.Text);
        }

        private void btnOpenSaveDir_Click(object sender, EventArgs e)
        {
            FileUtils.OpenFolderAndSelectFile(txtSavePath.Text);
        }


        // ---------------------------------------------------------
        // 功能按鈕
        // ---------------------------------------------------------
        private void btnBrighten_Click(object sender, EventArgs e)
        {
            RunProcessGPU((src, dst, w, h) =>
                TanukiCvWrapper.TanukiCv_Brighten_GPU(src, w, h, (int)numBrightVal.Value, dst),
                lblBrightenTime
            );
        }

        private void btnBinary_Click(object sender, EventArgs e)
        {
            RunProcessGPU((src, dst, w, h) =>
                TanukiCvWrapper.TanukiCv_Threshold_GPU(src, w, h, (byte)numThreshold.Value, dst),
                lblBinaryTime
            );
        }

        private void btnInvert_Click(object sender, EventArgs e)
        {
            RunProcessGPU((src, dst, w, h) =>
                TanukiCvWrapper.TanukiCv_Invert_GPU(src, w, h, dst),
                lblInvertTime
            );
        }

        private void btnConvolution_Click(object sender, EventArgs e)
        {
            // 定義 Mask (銳化)
            float[] mask = {
                 0, -1,  0,
                -1,  5, -1,
                 0, -1,  0
            };

            // 取得 GPU 上的 Mask 指標 (GPUProcessor 會負責快取，只在第一次上傳)
            IntPtr d_mask = _gpu.GetConvolutionMask(mask);

            // 執行卷積
            RunProcessGPU((src, dst, w, h) =>
                TanukiCvWrapper.TanukiCv_Convolution_GPU(src, w, h, d_mask, 3, dst),
                lblConvTime
            );
        }

        // ---------------------------------------------------------
        // 核心處理邏輯 (GPU Resident Version)
        // ---------------------------------------------------------
        delegate int DllFuncGPU(IntPtr d_src, IntPtr d_dst, int w, int h);

        private void RunProcessGPU(GPUKernelFunc func, Label targetLabel)
        {
            if (!_gpu.IsInitialized) { MessageBox.Show("請先讀取圖片"); return; }

            // 1. 執行運算 (Processor 內部會計時並交換 Ping-Pong 指標)
            int err;
            double ms = _gpu.Process(func, out err);

            // 更新 UI 時間
            targetLabel.Text = $"{ms:F3} ms";

            if (err == 0) // Success
            {
                // 2. 下載結果回 _currentData (Host)
                _gpu.DownloadResult(_currentData);

                // 3. 顯示圖片
                ShowImage(_currentData);
            }
            else
            {
                MessageBox.Show($"GPU Error Code: {err}");
            }
        }

        // ---------------------------------------------------------
        // 顯示與轉檔輔助
        // ---------------------------------------------------------
        private void ShowImage(byte[] data)
        {
            if (_imgW <= 0 || _imgH <= 0) return;

            // 使用 ImageUtils 生成 Bitmap
            var bmp = ImageUtils.Create8bppBitmap(data, _imgW, _imgH);

            // 更新 Canvas，並釋放舊圖
            var oldImg = canvasMain.Image;
            canvasMain.Image = bmp;
            if (oldImg != null) oldImg.Dispose();
        }


    }
}