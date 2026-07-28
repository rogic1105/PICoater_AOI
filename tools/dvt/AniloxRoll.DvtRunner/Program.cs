using System;
using System.IO;
using System.Windows.Forms;

namespace AniloxRoll.DvtRunner
{
    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            string scenarioId = ReadArgument(args, "--scenario");
            string resultPath = ReadArgument(args, "--result-file");
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm(scenarioId, resultPath));
        }

        private static string ReadArgument(string[] args, string name)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (!string.Equals(
                    args[i], name, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (i + 1 >= args.Length)
                    throw new InvalidDataException(
                        name + " requires a value.");
                return args[i + 1];
            }
            return null;
        }
    }
}
