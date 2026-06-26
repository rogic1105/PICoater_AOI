using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace TanukiCv.BenchUi.Utils
{
    /// <summary>檔案/資料夾 explorer 開啟工具（WinForms：含 MessageBox 錯誤提示）。
    /// 原在 TanukiCv.Core（純 library 不該帶 WinForms）→ 移到唯一使用者 BenchUi，讓 Core 保持純淨。</summary>
    public static class FileUtils
    {
        public static void OpenFolderAndSelectFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath)) return;

            try
            {
                if (File.Exists(filePath))
                {
                    Process.Start("explorer.exe", "/select,\"" + filePath + "\"");
                }
                else
                {
                    string dir = Path.GetDirectoryName(filePath);
                    if (Directory.Exists(dir))
                    {
                        Process.Start("explorer.exe", "\"" + dir + "\"");
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("無法開啟資料夾: " + ex.Message);
            }
        }
    }
}
