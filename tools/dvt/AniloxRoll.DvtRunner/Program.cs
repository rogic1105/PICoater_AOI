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
            string processIdPath = ReadArgument(args, "--process-id-file");
            int? durationSeconds = ReadPositiveIntArgument(
                args, "--duration-seconds");
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm(
                scenarioId, resultPath, processIdPath, durationSeconds));
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

        private static int? ReadPositiveIntArgument(
            string[] args,
            string name)
        {
            string value = ReadArgument(args, name);
            if (value == null) return null;

            int parsed;
            if (!int.TryParse(value, out parsed) || parsed <= 0)
                throw new InvalidDataException(
                    name + " requires a positive integer.");
            return parsed;
        }
    }
}
