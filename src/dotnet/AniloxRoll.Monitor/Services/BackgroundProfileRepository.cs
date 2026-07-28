using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    internal enum BackgroundManifestStatus
    {
        Missing,
        Active,
        Invalid
    }

    internal sealed class BackgroundManifestSnapshot
    {
        public BackgroundManifestSnapshot(
            BackgroundManifestStatus status,
            string version)
        {
            Status = status;
            Version = version;
        }

        public BackgroundManifestStatus Status { get; }
        public string Version { get; }
    }

    /// <summary>
    /// Owns persisted camera background profiles and the atomic active-version manifest.
    /// Product workflow and health policy remain with the application coordinator.
    /// </summary>
    internal sealed class BackgroundProfileRepository
    {
        private const string InvalidActiveProfileName =
            "__invalid-active-background__.bin";

        public BackgroundProfileRepository(string rootPath)
        {
            RootPath = rootPath;
        }

        public string RootPath { get; }

        public bool DirectoryExists =>
            !string.IsNullOrWhiteSpace(RootPath) &&
            Directory.Exists(RootPath);

        public void EnsureDirectory()
        {
            if (string.IsNullOrWhiteSpace(RootPath))
                throw new IOException("背景目錄未設定");
            Directory.CreateDirectory(RootPath);
        }

        public string SaveCameraProfile(
            float[] data,
            int width,
            int cameraId,
            string version,
            int lightLevel,
            float exposureUs)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (data.Length != width)
                throw new ArgumentException("背景資料長度與影像寬度不一致", nameof(data));
            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("背景版本不得為空", nameof(version));

            EnsureDirectory();
            string path = Path.Combine(
                RootPath,
                CaptureFileNaming.BgVersionedBin(width, cameraId, version));

            using (var stream = new FileStream(
                path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096,
                FileOptions.WriteThrough))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                writer.Write(2);
                writer.Write(1.0f);
                writer.Write(lightLevel);
                writer.Write(exposureUs);
                writer.Write(data.Length);
                foreach (float value in data)
                    writer.Write(value);
                writer.Flush();
                stream.Flush(true);
            }

            return path;
        }

        public void ActivateVersion(string version)
        {
            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("背景版本不得為空", nameof(version));

            EnsureDirectory();
            string manifest = Path.Combine(
                RootPath, CaptureFileNaming.BgActiveManifest);
            string temp = manifest + ".tmp-" + Guid.NewGuid().ToString("N");
            try
            {
                byte[] bytes = Encoding.UTF8.GetBytes(
                    "{\r\n  \"Version\": \"" + version + "\"\r\n}");
                using (var stream = new FileStream(
                    temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096,
                    FileOptions.WriteThrough))
                {
                    stream.Write(bytes, 0, bytes.Length);
                    stream.Flush(true);
                }

                if (File.Exists(manifest))
                    File.Replace(temp, manifest, null, true);
                else
                    File.Move(temp, manifest);
            }
            finally
            {
                TryDelete(temp);
            }
        }

        public BackgroundManifestSnapshot ReadManifest()
        {
            if (!DirectoryExists)
            {
                return new BackgroundManifestSnapshot(
                    BackgroundManifestStatus.Missing, null);
            }

            string manifest = Path.Combine(
                RootPath, CaptureFileNaming.BgActiveManifest);
            if (!File.Exists(manifest))
            {
                return new BackgroundManifestSnapshot(
                    BackgroundManifestStatus.Missing, null);
            }

            try
            {
                string json = File.ReadAllText(manifest, Encoding.UTF8);
                string version = SettingsStoreHelper.GetString(
                    json, "Version", null);
                if (!string.IsNullOrWhiteSpace(version))
                {
                    return new BackgroundManifestSnapshot(
                        BackgroundManifestStatus.Active, version);
                }
            }
            catch
            {
            }

            return new BackgroundManifestSnapshot(
                BackgroundManifestStatus.Invalid, null);
        }

        public string ResolveCameraProfilePath(int width, int cameraId)
        {
            BackgroundManifestSnapshot manifest = ReadManifest();
            if (manifest.Status == BackgroundManifestStatus.Active)
            {
                return Path.Combine(
                    RootPath,
                    CaptureFileNaming.BgVersionedBin(
                        width, cameraId, manifest.Version));
            }

            if (manifest.Status == BackgroundManifestStatus.Invalid)
                return Path.Combine(RootPath, InvalidActiveProfileName);

            return Path.Combine(
                RootPath, CaptureFileNaming.BgBin(width, cameraId));
        }

        public string ResolvePreviewProfilePath(int cameraId)
        {
            if (!DirectoryExists) return null;

            BackgroundManifestSnapshot manifest = ReadManifest();
            if (manifest.Status == BackgroundManifestStatus.Invalid)
                return null;

            string pattern = manifest.Status == BackgroundManifestStatus.Active
                ? CaptureFileNaming.BgVersionedGlobForCam(
                    cameraId, manifest.Version)
                : CaptureFileNaming.BgGlobForCam(cameraId);
            string[] matches = Directory.GetFiles(RootPath, pattern);
            return matches.Length == 0 ? null : matches[0];
        }

        public float[] LoadProfile(string path)
        {
            return CurveBinFile.Load(path);
        }

        public bool HasAnyProfile()
        {
            if (!DirectoryExists) return false;

            BackgroundManifestSnapshot manifest = ReadManifest();
            if (manifest.Status == BackgroundManifestStatus.Invalid)
                return false;

            string pattern = manifest.Status == BackgroundManifestStatus.Active
                ? CaptureFileNaming.BgVersionedGlobForCam(1, manifest.Version)
                : CaptureFileNaming.BgGlob;
            return Directory.GetFiles(RootPath, pattern).Length > 0;
        }

        public void DeleteVersion(string version)
        {
            if (!DirectoryExists || string.IsNullOrWhiteSpace(version)) return;

            foreach (string path in Directory.GetFiles(
                RootPath, "bg_*_" + version + ".bin"))
            {
                TryDelete(path);
            }
        }

        public void CleanupInactiveVersions()
        {
            if (!DirectoryExists) return;

            BackgroundManifestSnapshot manifest = ReadManifest();
            if (manifest.Status != BackgroundManifestStatus.Active) return;

            foreach (string path in Directory.GetFiles(
                RootPath, CaptureFileNaming.BgGlob))
            {
                string name = Path.GetFileNameWithoutExtension(path);
                if (name.EndsWith(
                    "_" + manifest.Version,
                    StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                try
                {
                    File.Delete(path);
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning(
                        $"[BackgroundRetention] delete failed {path}: {ex.Message}");
                }
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch
            {
            }
        }
    }
}
