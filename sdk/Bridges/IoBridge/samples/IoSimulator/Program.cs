using System;
using System.Windows.Forms;

namespace IoBridge.IoSimulator
{
    internal static class Program
    {
        [STAThread]
        static int Main(string[] args)
        {
            if (AutomatedCycleRunner.IsRequested(args))
                return AutomatedCycleRunner.Run(args);

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
            return 0;
        }
    }
}
