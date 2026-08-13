namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// PowerShell-facing facade for one-time archive maintenance. Runtime capture and review
    /// use CaptureArchiveStore and never depend on this command surface.
    /// </summary>
    public static class CaptureArchiveMigration
    {
        public static string ConvertLegacyRoot(string captureRoot, bool overwrite)
        {
            CaptureArchiveConversionResult result = CaptureArchiveLegacyConverter.ConvertRoot(
                captureRoot, overwrite);
            return string.Format(
                "archives={0};frames={1};payloadBytes={2};skipped={3};failed={4}",
                result.ArchiveCount,
                result.FrameCount,
                result.PayloadBytes,
                result.SkippedArchiveCount,
                result.FailedArchiveCount);
        }

        public static string ValidateRoot(string captureRoot)
        {
            CaptureArchiveValidationResult result = CaptureArchiveStore.ValidateRoot(captureRoot);
            return string.Format(
                "archives={0};rawFrames={1};records={2};payloadBytes={3};" +
                "previewAtlases={4};invalidArchives={5};invalidRecords={6};partialFiles={7}",
                result.ArchiveCount,
                result.RawFrameCount,
                result.RecordCount,
                result.PayloadBytes,
                result.PreviewAtlasCount,
                result.InvalidArchiveCount,
                result.InvalidRecordCount,
                result.PartialFileCount);
        }

        public static string AddThumbnails(string captureRoot, int targetWidth)
        {
            CaptureArchiveThumbnailResult result =
                CaptureArchiveThumbnailMaintenance.AddThumbnails(captureRoot, targetWidth);
            return string.Format(
                "archives={0};frames={1};thumbnails={2};thumbnailBytes={3};" +
                "skipped={4};failed={5}",
                result.ArchiveCount,
                result.FrameCount,
                result.ThumbnailCount,
                result.ThumbnailBytes,
                result.SkippedThumbnailCount,
                result.FailedFrameCount);
        }

        public static string AddPreviewAtlases(
            string captureRoot, int maxWidth, int maxHeight,
            bool replaceExisting)
        {
            CaptureArchivePreviewAtlasResult result =
                CapturePreviewAtlasCodec.AddToRoot(
                    captureRoot, maxWidth, maxHeight,
                    replaceExisting, null);
            return string.Format(
                "archives={0};atlases={1};atlasBytes={2};skipped={3};failed={4}",
                result.ArchiveCount,
                result.AtlasCount,
                result.AtlasBytes,
                result.SkippedAtlasCount,
                result.FailedArchiveCount);
        }

        public static string AddPreviewAtlases(
            string captureRoot, int maxWidth, int maxHeight)
        {
            return AddPreviewAtlases(
                captureRoot, maxWidth, maxHeight, false);
        }
    }
}
