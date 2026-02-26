using System.IO;
using System.Xml.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 的序列化儲存層。
    /// 負責 XML 載入/儲存，並在載入時套用必要的安全預設值。
    /// </summary>
    public static class InspectionSettingsStore
    {
        /// <summary>
        /// 從使用者設定載入 InspectionSettings。
        /// </summary>
        public static InspectionSettings Load()
        {
            try
            {
                string xml = UserSettingsService.InspectionConfigJson;
                if (string.IsNullOrWhiteSpace(xml)) return new InspectionSettings();

                XmlSerializer serializer = new XmlSerializer(typeof(InspectionSettings));
                using (StringReader reader = new StringReader(xml))
                {
                    var settings = (InspectionSettings)serializer.Deserialize(reader);
                    if (settings.CameraGrabHeight <= 0) settings.CameraGrabHeight = 5000;
                    if (settings.CameraExposureTimeUs <= 0) settings.CameraExposureTimeUs = 50;
                    return settings;
                }
            }
            catch
            {
                return new InspectionSettings();
            }
        }

        /// <summary>
        /// 將 InspectionSettings 序列化後寫回使用者設定。
        /// </summary>
        public static void Save(InspectionSettings settings)
        {
            try
            {
                XmlSerializer serializer = new XmlSerializer(typeof(InspectionSettings));
                using (StringWriter writer = new StringWriter())
                {
                    serializer.Serialize(writer, settings);
                    UserSettingsService.InspectionConfigJson = writer.ToString();
                    UserSettingsService.Save();
                }
            }
            catch
            {
                // Ignore save error
            }
        }
    }
}
