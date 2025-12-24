using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Windows.Forms;

namespace AOI_GUI
{
    public partial class Form1 : Form
    {
        // ---------------------------------------------------------
        // 1. DLL 匯入 (對應 C++ 的 export_api.cpp)
        // ---------------------------------------------------------
        // 請確保 ExportCDLL.dll 在執行檔旁邊，或設定環境變數
        const string DLL_NAME = "ExportCDLL.dll";

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Process_Brighten(IntPtr src, IntPtr dst, int W, int H, int value);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Process_Threshold(IntPtr src, IntPtr dst, int W, int H, int value);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Process_Invert(IntPtr src, IntPtr dst, int W, int H);

        // ---------------------------------------------------------
        // 2. 變數宣告
        // ---------------------------------------------------------
        private byte[] _originalData; // 原始圖片數據 (8bpp)
        private byte[] _currentData;  // 目前顯示的圖片數據
        private int _imgW, _imgH;

        public Form1()
        {
            InitializeComponent();

            // 綁定 SmartCanvas 的滑鼠事件
            canvasMain.PixelHovered += (x, y, color) =>
            {
                lblPixelInfo.Text = $"座標: ({x}, {y}) | 亮度: {color.R}"; // 灰階圖 RGB 都一樣
            };
        }

        // ---------------------------------------------------------
        // 3. 讀取圖片
        // ---------------------------------------------------------
        private void btnLoad_Click(object sender, EventArgs e)
        {
            // 這一行必須要有，不然電腦不知道 ofd 是什麼
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image Files|*.bmp;*.png;*.jpg;*.tif";

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                using (Bitmap bmp = new Bitmap(ofd.FileName))
                {
                    // 強制轉為 8bpp 灰階
                    _originalData = ConvertTo8bppArray(bmp, out _imgW, out _imgH);

                    // 初始化目前數據
                    _currentData = new byte[_originalData.Length];
                    Array.Copy(_originalData, _currentData, _originalData.Length);

                    // 顯示圖片
                    ShowImage(_currentData);

                    // [新增功能] 載入後自動縮放適應視窗
                    // 確保你有在 SmartCanvas.cs 加入 FitToScreen() 函式
                    canvasMain.FitToScreen();
                }
            }
        }
        // ---------------------------------------------------------
        // 4. 功能按鈕 (每次都從原圖計算，單一操作)
        // ---------------------------------------------------------

        // 二值化
        private void btnBinary_Click(object sender, EventArgs e)
        {
            RunProcess((src, dst, w, h) =>
                Process_Threshold(src, dst, w, h, (int)numThreshold.Value)
            );
        }

        // 反轉
        private void btnInvert_Click(object sender, EventArgs e)
        {
            RunProcess((src, dst, w, h) =>
                Process_Invert(src, dst, w, h)
            );
        }

        // 變亮
        private void btnBrighten_Click(object sender, EventArgs e)
        {
            RunProcess((src, dst, w, h) =>
                Process_Brighten(src, dst, w, h, (int)numBrightVal.Value)
            );
        }

        // 回到原圖
        private void btnReset_Click(object sender, EventArgs e)
        {
            if (_originalData == null) return;
            // 把原圖覆蓋回當前圖
            Array.Copy(_originalData, _currentData, _originalData.Length);
            ShowImage(_currentData);
            // canvasMain.FitToScreen(); // 選擇性：Reset 時要不要也重置視角？看你習慣
        }

        // ---------------------------------------------------------
        // 5. 核心處理邏輯 (包裝指針操作)
        // ---------------------------------------------------------

        // 定義一個委派來簡化 DLL 呼叫
        delegate int DllFunc(IntPtr src, IntPtr dst, int w, int h);

        private void RunProcess(DllFunc func)
        {
            if (_originalData == null) { MessageBox.Show("請先讀取圖片"); return; }

            // [修改] 輸入與輸出都指向 _currentData
            // 這樣下一次操作時，輸入就會是上一次的結果
            GCHandle hSrc = GCHandle.Alloc(_currentData, GCHandleType.Pinned);
            GCHandle hDst = GCHandle.Alloc(_currentData, GCHandleType.Pinned); // Src == Dst

            try
            {
                IntPtr ptrSrc = hSrc.AddrOfPinnedObject();
                IntPtr ptrDst = hDst.AddrOfPinnedObject();

                int ret = func(ptrSrc, ptrDst, _imgW, _imgH);

                if (ret == 0)
                {
                    ShowImage(_currentData);
                }
                else
                {
                    MessageBox.Show($"DLL 執行錯誤，代碼: {ret}");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("發生錯誤: " + ex.Message);
            }
            finally
            {
                hSrc.Free();
                hDst.Free();
            }
        }

        // 顯示圖片 (Byte Array -> Bitmap)
        private void ShowImage(byte[] data)
        {
            Bitmap bmp = new Bitmap(_imgW, _imgH, PixelFormat.Format8bppIndexed);

            // 設定灰階調色盤
            ColorPalette pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            // 填入數據
            BitmapData bData = bmp.LockBits(new Rectangle(0, 0, _imgW, _imgH),
                ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);

            // 逐行複製 (處理 Stride)
            for (int y = 0; y < _imgH; y++)
            {
                Marshal.Copy(data, y * _imgW, bData.Scan0 + y * bData.Stride, _imgW);
            }

            bmp.UnlockBits(bData);

            canvasMain.Image = bmp;
            // canvasMain.ResetView(); // 如果不想每次重置縮放，這行可以註解掉
        }

        // 轉檔輔助：任意圖片 -> 8bpp Byte Array
        private byte[] ConvertTo8bppArray(Bitmap src, out int w, out int h)
        {
            w = src.Width;
            h = src.Height;
            byte[] result = new byte[w * h];

            // 轉成 24bpp 方便讀取 RGB
            using (Bitmap bmp24 = new Bitmap(w, h, PixelFormat.Format24bppRgb))
            {
                using (Graphics g = Graphics.FromImage(bmp24))
                {
                    g.DrawImage(src, 0, 0, w, h);
                }

                BitmapData bData = bmp24.LockBits(new Rectangle(0, 0, w, h),
                    ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

                int stride = bData.Stride;
                byte[] rgbData = new byte[stride * h];
                Marshal.Copy(bData.Scan0, rgbData, 0, rgbData.Length);
                bmp24.UnlockBits(bData);

                // RGB -> Gray
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int idx = y * stride + x * 3;
                        byte B = rgbData[idx];
                        byte G = rgbData[idx + 1];
                        byte R = rgbData[idx + 2];
                        // 簡單平均
                        result[y * w + x] = (byte)((R + G + B) / 3);
                    }
                }
            }
            return result;
        }
    }
}