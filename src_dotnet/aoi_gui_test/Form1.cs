using AOIUI.Interop;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace CsTestApp
{
    public partial class Form1 : Form
    {

        private byte[] _srcTight;   // 緊密 (pitch == W) 的灰階資料
        private int _w, _h;

        public Form1()
        {
            InitializeComponent();

            // 如果 DLL 不在輸出資料夾，可用這行把資料夾加進 DLL 搜尋路徑
            //Win32.SetDllDirectory(@"D:\Chunkuan\ArrayAOI\cpp\test\x64\Debug");
        }

        private void btnOpen_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.bmp;*.tif;*.tiff" })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;

                using (var bmpAny = new Bitmap(ofd.FileName))
                {
                    Bitmap gray = To8bppTightGray(bmpAny, out _w, out _h, out _srcTight);
                    if (pictureBoxSrc.Image != null) pictureBoxSrc.Image.Dispose();
                    pictureBoxSrc.Image = gray;
                    if (pictureBoxDst.Image != null) pictureBoxDst.Image.Dispose();
                    pictureBoxDst.Image = null;
                }
            }
        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            if (_srcTight == null)
            {
                MessageBox.Show("請先開啟圖片");
                return;
            }

            var fft_th = (int)numfft_th.Value;
            var bw_th = (byte)numbw_th.Value;
            var border_t = (byte)numborder_t.Value;

            var outBuf = new byte[_srcTight.Length];
            int err = AOILib.BuildFullMaskFromFFT_C(_srcTight, _h, _w, fft_th, bw_th, border_t, outBuf);
            if (err != 0)
            {
                MessageBox.Show($"A_BrightenThenThreshold 失敗，err={err}");
                return;
            }

            // 將緊密灰階資料轉成可顯示的 8bpp Bitmap（注意 GDI+ stride 對齊）
            var outBmp = FromTightGrayTo8bppBitmap(outBuf, _w, _h);
            pictureBoxDst.Image?.Dispose();
            pictureBoxDst.Image = outBmp;
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            if (pictureBoxDst.Image == null)
            {
                MessageBox.Show("沒有結果可以存檔");
                return;
            }

            using (var sfd = new SaveFileDialog())
            {
                sfd.Filter = "PNG|*.png";
                sfd.AddExtension = true;
                sfd.DefaultExt = "png";

                if (sfd.ShowDialog() != DialogResult.OK)
                    return;

                try
                {
                    pictureBoxDst.Image.Save(sfd.FileName, ImageFormat.Png);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("存檔失敗：\n" + ex.Message);
                }
            }
        }

        // 任何輸入 Bitmap → 轉「緊密」灰階（W*H），同時回傳一張 8bpp 灰階 Bitmap 方便預覽
        private static Bitmap To8bppTightGray(Bitmap src, out int w, out int h, out byte[] tight)
        {
            w = src.Width; h = src.Height;
            tight = new byte[w * h];

            using (var rgb = new Bitmap(w, h, PixelFormat.Format24bppRgb))
            {
                using (var g = Graphics.FromImage(rgb))
                    g.DrawImage(src, new Rectangle(0, 0, w, h));

                var rect = new Rectangle(0, 0, w, h);
                var data = rgb.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    int stride = data.Stride;
                    var rowBuf = new byte[stride];

                    for (int y = 0; y < h; y++)
                    {
                        Marshal.Copy(data.Scan0 + y * stride, rowBuf, 0, stride);
                        int ti = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            byte b = rowBuf[x * 3 + 0];
                            byte g = rowBuf[x * 3 + 1];
                            byte r = rowBuf[x * 3 + 2];
                            int v = (int)(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
                            if (v < 0) v = 0; else if (v > 255) v = 255;
                            tight[ti + x] = (byte)v;
                        }
                    }
                }
                finally { rgb.UnlockBits(data); }
            }

            return FromTightGrayTo8bppBitmap(tight, w, h);
        }
        // 緊密灰階（W*H）→ 8bpp Bitmap（GDI+ 的 stride 會是 4 對齊）
        private static Bitmap FromTightGrayTo8bppBitmap(byte[] tight, int w, int h)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);

            // 灰階調色盤
            var pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            var rect = new Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            try
            {
                int stride = data.Stride;
                for (int y = 0; y < h; y++)
                {
                    Marshal.Copy(tight, y * w, data.Scan0 + y * stride, w);
                }
            }
            finally { bmp.UnlockBits(data); }

            return bmp;
        }
    }
}
