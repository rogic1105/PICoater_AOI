using System;
using System.IO;
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

        public void LoadDirectoryAndInitNavigator(string path)
        {
            _imageRepository.LoadDirectory(path);
            FlowTrace.Log($"RV repo scan root={path} files={_imageRepository.FileCount}");
            if (_imageRepository.FileCount > 0)
                _timeNavigator.Initialize(UserSessionState.LastYear);
        }

        public void SelectAndLoadFolder()
        {
            using (var dialog = new FolderBrowserDialog())
            {
                string preferredPath = UserSessionState.LastDataPath;
                if (!Directory.Exists(preferredPath)) preferredPath = _settings?.CaptureRootPath;
                if (string.IsNullOrEmpty(preferredPath) || !Directory.Exists(preferredPath))
                    preferredPath = Path.Combine(
                        InspectionDefaults.AniloxRootPath,
                        InspectionDefaults.CaptureDirectoryName);
                if (Directory.Exists(preferredPath))
                    dialog.SelectedPath = preferredPath;

                if (dialog.ShowDialog() != DialogResult.OK) return;

                string selectedPath = dialog.SelectedPath;
                if (!HasYearSubdir(selectedPath))
                {
                    string capturesSub = Path.Combine(
                        selectedPath, InspectionDefaults.CaptureDirectoryName);
                    if (HasYearSubdir(capturesSub)) selectedPath = capturesSub;
                }

                UserSessionState.SetLastDataPath(selectedPath);
                UserSessionState.Save();

                FlowTrace.Log($"RV folder selected root={selectedPath}");
                _imageRepository.LoadDirectory(selectedPath);
                FlowTrace.Log($"RV repo scan root={selectedPath} files={_imageRepository.FileCount}");
                if (_imageRepository.FileCount == 0)
                {
                    MessageBox.Show(_dialogOwner, "該路徑下無符合格式的圖片！");
                    return;
                }

                _timeNavigator.Initialize(UserSessionState.LastYear);
            }
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
