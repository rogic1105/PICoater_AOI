using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Managers;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Coordinates one background sampling run after cameras are already allocated and grabbing.
    /// Light, buttons, output-health policy, stopping, and preview remain with the Form.
    /// </summary>
    internal sealed class BackgroundCaptureCoordinator
    {
        private readonly LiveCameraManager _cameraManager;
        private readonly BackgroundProfileRepository _repository;

        public BackgroundCaptureCoordinator(
            LiveCameraManager cameraManager,
            BackgroundProfileRepository repository)
        {
            _cameraManager = cameraManager ??
                throw new ArgumentNullException(nameof(cameraManager));
            _repository = repository ??
                throw new ArgumentNullException(nameof(repository));
        }

        public async Task CaptureAndActivateAsync(
            int sampleSeconds,
            int lightLevel,
            Action<int> remainingSecondsChanged)
        {
            if (sampleSeconds < 1)
                throw new ArgumentOutOfRangeException(nameof(sampleSeconds));

            string version = DateTime.Now.ToString("yyyyMMdd-HHmmssfff");
            try
            {
                await WaitForFirstSetAsync();
                await SampleAndSaveAsync(
                    sampleSeconds,
                    lightLevel,
                    version,
                    remainingSecondsChanged);
                _repository.ActivateVersion(version);
            }
            catch
            {
                _repository.DeleteVersion(version);
                throw;
            }
        }

        private async Task WaitForFirstSetAsync()
        {
            int timeoutMs = _cameraManager.GetCaptureFirstSetTimeoutMs();
            FlowTrace.Log(
                $"background capture waiting first-set timeoutMs={timeoutMs}");
            bool ready = await _cameraManager.WaitForCaptureFirstSetReadyAsync(
                timeoutMs);
            if (!ready)
            {
                throw new IOException(
                    $"背景採樣未等到完整首幀組 ({timeoutMs}ms)");
            }
        }

        private async Task SampleAndSaveAsync(
            int sampleSeconds,
            int lightLevel,
            string version,
            Action<int> remainingSecondsChanged)
        {
            FlowTrace.Log(
                $"background capture sampling start duration={sampleSeconds}s");
            _repository.EnsureDirectory();

            IReadOnlyList<AniloxCamera> cameras = _cameraManager.Cameras;
            int cameraCount = cameras.Count;
            var accumulated = new double[cameraCount][];
            var frameCounts = new int[cameraCount];

            var stopwatch = Stopwatch.StartNew();
            int lastRemaining = -1;
            while (stopwatch.Elapsed.TotalSeconds < sampleSeconds)
            {
                int remaining =
                    sampleSeconds - (int)stopwatch.Elapsed.TotalSeconds;
                if (remaining != lastRemaining)
                {
                    lastRemaining = remaining;
                    remainingSecondsChanged?.Invoke(remaining);
                }

                await Task.Delay(100);
                SampleCurrentFrames(cameras, accumulated, frameCounts);
            }

            FlowTrace.Log(
                $"background capture sampling complete durationMs={stopwatch.ElapsedMilliseconds} " +
                $"frames={string.Join(",", cameras.Select((camera, index) => $"cam{camera.CameraId}:{frameCounts[index]}"))}");

            int savedCameraCount = SaveAverages(
                cameras,
                accumulated,
                frameCounts,
                lightLevel,
                version);
            if (savedCameraCount == 0)
                throw new IOException("沒有任何相機產生背景檔");
        }

        private static void SampleCurrentFrames(
            IReadOnlyList<AniloxCamera> cameras,
            double[][] accumulated,
            int[] frameCounts)
        {
            for (int cameraIndex = 0;
                cameraIndex < cameras.Count;
                cameraIndex++)
            {
                AniloxCamera camera = cameras[cameraIndex];
                if (!camera.IsConnected || camera.FrameWidth <= 0) continue;

                if (accumulated[cameraIndex] == null)
                    accumulated[cameraIndex] = new double[camera.FrameWidth];

                var columnMean = new float[camera.FrameWidth];
                if (!camera.TryComputeColumnMean(columnMean)) continue;

                for (int column = 0; column < camera.FrameWidth; column++)
                    accumulated[cameraIndex][column] += columnMean[column];
                frameCounts[cameraIndex]++;
            }
        }

        private int SaveAverages(
            IReadOnlyList<AniloxCamera> cameras,
            double[][] accumulated,
            int[] frameCounts,
            int lightLevel,
            string version)
        {
            int savedCameraCount = 0;
            for (int cameraIndex = 0;
                cameraIndex < cameras.Count;
                cameraIndex++)
            {
                AniloxCamera camera = cameras[cameraIndex];
                if (!camera.IsConnected || camera.FrameWidth <= 0) continue;
                if (frameCounts[cameraIndex] == 0 ||
                    accumulated[cameraIndex] == null)
                {
                    throw new IOException(
                        $"CAM{camera.CameraId} 沒有取得有效背景樣本");
                }

                var average = new float[camera.FrameWidth];
                double inverseCount = 1.0 / frameCounts[cameraIndex];
                for (int column = 0; column < camera.FrameWidth; column++)
                {
                    average[column] =
                        (float)(accumulated[cameraIndex][column] * inverseCount);
                }

                _repository.SaveCameraProfile(
                    average,
                    camera.FrameWidth,
                    camera.CameraId,
                    version,
                    lightLevel,
                    (float)camera.CameraExposureTimeUs);
                savedCameraCount++;
            }

            return savedCameraCount;
        }
    }
}
