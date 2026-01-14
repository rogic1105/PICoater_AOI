namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// [Model] 影像檔案的詮釋資料模型。
    /// 解析檔名後將時間戳記 (Year, Month...) 與相機 ID 結構化儲存，
    /// 用於後續的快速查詢與篩選，避免重複解析字串。
    /// </summary>
    public class ImageMetadata
    {
        public string FullPath { get; set; }
        public string Year { get; set; }
        public string Month { get; set; }
        public string Day { get; set; }
        public string Hour { get; set; }
        public string Minute { get; set; }
        public string Second { get; set; }
        public int CameraId { get; set; }
    }
}