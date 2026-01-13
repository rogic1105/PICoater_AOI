// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Program.cs

using System;
using System.Windows.Forms;
using AniloxRoll.Monitor.Forms; // [修改]

namespace AniloxRoll.Monitor
{
    internal static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            // [修改] 啟動 MainForm
            Application.Run(new AniloxRollForm());
        }
    }
}