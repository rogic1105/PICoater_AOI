using System;
using System.IO;

namespace AniloxRoll.DvtRunner
{
    internal static class RepositoryLocator
    {
        public static string FindRoot()
        {
            string fromBase = WalkUp(AppDomain.CurrentDomain.BaseDirectory);
            if (fromBase != null) return fromBase;

            string fromCurrent = WalkUp(Environment.CurrentDirectory);
            if (fromCurrent != null) return fromCurrent;

            throw new DirectoryNotFoundException(
                "Cannot locate PICoater_AOI.sln. Start the runner from a repository build.");
        }

        private static string WalkUp(string start)
        {
            var directory = new DirectoryInfo(Path.GetFullPath(start));
            while (directory != null)
            {
                if (File.Exists(Path.Combine(directory.FullName, "PICoater_AOI.sln")))
                    return directory.FullName;
                directory = directory.Parent;
            }
            return null;
        }
    }
}
