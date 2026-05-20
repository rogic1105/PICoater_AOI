using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    internal static class AppPaths
    {
        public static readonly string ConfigDir =
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
                "AniloxRoll", "Config");
    }
}
