using System.IO;
using System.Xml.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    public static class InspectionSettingsStore
    {
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
