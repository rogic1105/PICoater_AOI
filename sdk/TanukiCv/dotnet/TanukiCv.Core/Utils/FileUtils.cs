// TanukiCv.Core\Utils\FileUtils.cs

using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;


namespace TanukiCv.Utils
{
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