using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;

namespace IoBridge.IoSimulator
{
    internal static class AutomatedCycleRunner
    {
        public static bool IsRequested(string[] args)
        {
            foreach (string arg in args ?? Array.Empty<string>())
            {
                if (string.Equals(arg, "--auto", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        public static int Run(string[] args)
        {
            Options options = null;
            var evidence = new List<string>();
            ModbusTcpServer server = null;
            try
            {
                options = Options.Parse(args);
                server = new ModbusTcpServer();
                server.Log = message => Record(evidence, message);
                server.SetDi(0, true);
                server.SetDi(1, false);
                server.Start(options.Port);

                Record(
                    evidence,
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "automation start port={0} cycles={1} initialMs={2} highMs={3} lowMs={4}",
                        options.Port,
                        options.Cycles,
                        options.InitialDelayMs,
                        options.HighMs,
                        options.LowMs));
                Thread.Sleep(options.InitialDelayMs);

                for (int cycle = 1; cycle <= options.Cycles; cycle++)
                {
                    Stopwatch high = Stopwatch.StartNew();
                    server.SetDi(1, true);
                    Thread.Sleep(options.HighMs);
                    high.Stop();
                    Record(
                        evidence,
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "cycle={0} HIGH actualMs={1} targetMs={2}",
                            cycle,
                            high.ElapsedMilliseconds,
                            options.HighMs));

                    Stopwatch low = Stopwatch.StartNew();
                    server.SetDi(1, false);
                    Thread.Sleep(options.LowMs);
                    low.Stop();
                    Record(
                        evidence,
                        string.Format(
                            CultureInfo.InvariantCulture,
                            "cycle={0} LOW actualMs={1} targetMs={2}",
                            cycle,
                            low.ElapsedMilliseconds,
                            options.LowMs));
                }

                Thread.Sleep(options.ExitDelayMs);
                Record(
                    evidence,
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "automation complete cycles={0} do0={1} do1={2} do2={3}",
                        options.Cycles,
                        server.GetDo(0),
                        server.GetDo(1),
                        server.GetDo(2)));
                WriteEvidence(options.ResultFile, evidence);
                return 0;
            }
            catch (Exception ex)
            {
                Record(evidence, "FAIL " + ex);
                if (options != null)
                    WriteEvidence(options.ResultFile, evidence);
                return 1;
            }
            finally
            {
                try { server?.Stop(); } catch { }
            }
        }

        private static void Record(ICollection<string> evidence, string message)
        {
            evidence.Add(DateTime.Now.ToString("O") + " " + message);
        }

        private static void WriteEvidence(string path, IEnumerable<string> evidence)
        {
            if (string.IsNullOrWhiteSpace(path))
                return;
            string fullPath = Path.GetFullPath(path);
            string directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllLines(fullPath, evidence, new UTF8Encoding(false));
        }

        private sealed class Options
        {
            public int Port { get; private set; } = 502;
            public int Cycles { get; private set; } = 1;
            public int InitialDelayMs { get; private set; } = 15000;
            public int HighMs { get; private set; } = 10000;
            public int LowMs { get; private set; } = 2000;
            public int ExitDelayMs { get; private set; } = 5000;
            public string ResultFile { get; private set; }

            public static Options Parse(string[] args)
            {
                var options = new Options();
                for (int i = 0; i < (args?.Length ?? 0); i++)
                {
                    string name = args[i];
                    if (string.Equals(name, "--auto", StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (i + 1 >= args.Length)
                        throw new ArgumentException("Missing value for " + name);
                    string value = args[++i];
                    switch (name.ToLowerInvariant())
                    {
                        case "--port":
                            options.Port = ParsePositiveInt(name, value);
                            break;
                        case "--cycles":
                            options.Cycles = ParsePositiveInt(name, value);
                            break;
                        case "--initial-delay-ms":
                            options.InitialDelayMs = ParseNonNegativeInt(name, value);
                            break;
                        case "--high-ms":
                            options.HighMs = ParsePositiveInt(name, value);
                            break;
                        case "--low-ms":
                            options.LowMs = ParsePositiveInt(name, value);
                            break;
                        case "--exit-delay-ms":
                            options.ExitDelayMs = ParseNonNegativeInt(name, value);
                            break;
                        case "--result-file":
                            options.ResultFile = value;
                            break;
                        default:
                            throw new ArgumentException("Unknown option " + name);
                    }
                }
                if (options.Port > 65535)
                    throw new ArgumentOutOfRangeException("--port");
                return options;
            }

            private static int ParsePositiveInt(string name, string value)
            {
                int parsed = ParseNonNegativeInt(name, value);
                if (parsed == 0)
                    throw new ArgumentOutOfRangeException(name);
                return parsed;
            }

            private static int ParseNonNegativeInt(string name, string value)
            {
                if (!int.TryParse(
                    value,
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out int parsed) ||
                    parsed < 0)
                    throw new ArgumentException(
                        name + " must be a non-negative integer.");
                return parsed;
            }
        }
    }
}
