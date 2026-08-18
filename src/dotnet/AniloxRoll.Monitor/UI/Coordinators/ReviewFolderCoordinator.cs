using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.State;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Owns review folder selection, repository refresh, and navigator initialization.
    /// </summary>
    public sealed class ReviewFolderCoordinator
    {
        private readonly IWin32Window _dialogOwner;
        private readonly ImageRepository _imageRepository;
        private readonly DateTimeNavigator _timeNavigator;
        private readonly InspectionSettings _settings;

        public ReviewFolderCoordinator(
            IWin32Window dialogOwner,
            ImageRepository imageRepository,
            DateTimeNavigator timeNavigator,
            InspectionSettings settings)
        {
            _dialogOwner = dialogOwner;
            _imageRepository = imageRepository ?? throw new ArgumentNullException(nameof(imageRepository));
            _timeNavigator = timeNavigator ?? throw new ArgumentNullException(nameof(timeNavigator));
            _settings = settings;
        }

        public void NavigateToDateTime(DateTime value) => _timeNavigator.NavigateTo(value);

        public async Task LoadDirectoryAndInitNavigatorAsync(string path)
        {
            FlowTrace.Log($"RV repo scan begin root={path}");
            ImageRepositoryLoadResult result =
                await _imageRepository.LoadDirectoryAsync(path);
            LogScanResult(path, result);
            if (_imageRepository.FileCount > 0)
                _timeNavigator.Initialize(UserSessionState.LastYear);
        }

        public async Task<bool> SelectAndLoadFolderAsync()
        {
            using (var dialog = new FolderBrowserDialog())
            {
                string configuredRoot = _settings?.CaptureRootPath;
                string sessionRoot = UserSessionState.LastDataPath;
                string preferredPath = CaptureStoragePaths.ResolveSelectedDataRoot(
                    sessionRoot,
                    configuredRoot);
                if (!string.Equals(
                    preferredPath,
                    sessionRoot,
                    StringComparison.OrdinalIgnoreCase))
                {
                    UserSessionState.SetLastDataPath(preferredPath);
                    UserSessionState.Save();
                    FlowTrace.Log(
                        $"RV data root upgraded from={sessionRoot} to={preferredPath}");
                }
                if (!Directory.Exists(preferredPath)) preferredPath = _settings?.CaptureRootPath;
                if (string.IsNullOrEmpty(preferredPath) || !Directory.Exists(preferredPath))
                    preferredPath = Path.Combine(
                        InspectionDefaults.AniloxRootPath,
                        InspectionDefaults.CaptureDirectoryName);
                if (Directory.Exists(preferredPath))
                    dialog.SelectedPath = preferredPath;

                if (dialog.ShowDialog() != DialogResult.OK) return false;

                string selectedPath = CaptureStoragePaths.ResolveSelectedDataRoot(
                    dialog.SelectedPath,
                    configuredRoot);
                if (!HasYearSubdir(selectedPath))
                {
                    string capturesSub = Path.Combine(
                        selectedPath, InspectionDefaults.CaptureDirectoryName);
                    if (HasYearSubdir(capturesSub)) selectedPath = capturesSub;
                }

                UserSessionState.SetLastDataPath(selectedPath);
                UserSessionState.Save();

                FlowTrace.Log($"RV folder selected root={selectedPath}");
                FlowTrace.Log($"RV repo scan begin root={selectedPath}");
                ImageRepositoryLoadResult result =
                    await _imageRepository.LoadDirectoryAsync(selectedPath);
                LogScanResult(selectedPath, result);
                if (_imageRepository.FileCount == 0)
                {
                    MessageBox.Show(_dialogOwner, "該路徑下無符合格式的圖片！");
                    return false;
                }

                _timeNavigator.Initialize(UserSessionState.LastYear);
                return true;
            }
        }

        private static void LogScanResult(
            string path, ImageRepositoryLoadResult result)
        {
            FlowTrace.Log(
                $"RV repo scan root={path} files={result.FileCount} " +
                $"csvRecords={result.CsvRecordCount} " +
                $"csvArchives={result.CsvBackedArchiveCount} " +
                $"archiveFallback={result.ArchiveFallbackCount} " +
                $"legacy={result.LegacyFileCount} " +
                $"enumMs={result.EnumerationMilliseconds} " +
                $"archiveIndexMs={result.ArchiveIndexMilliseconds} " +
                $"metadataMs={result.MetadataIndexMilliseconds} " +
                $"periodMs={result.PeriodIndexMilliseconds} " +
                $"ms={result.ElapsedMilliseconds}");
        }

        private static bool HasYearSubdir(string path)
        {
            if (string.IsNullOrEmpty(path) || !Directory.Exists(path)) return false;
            try
            {
                foreach (var directory in Directory.GetDirectories(path))
                {
                    string name = Path.GetFileName(directory);
                    if (name.Length == 4 && int.TryParse(name, out _)) return true;
                }
            }
            catch
            {
                // Permission and transient IO failures are treated as no matching directory.
            }

            return false;
        }
    }
}
