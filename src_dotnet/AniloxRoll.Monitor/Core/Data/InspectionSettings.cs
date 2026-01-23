using System;
using System.ComponentModel;
using System.IO;
using System.Xml.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    // [Serializable] 標記，讓它支援 XML 序列化
    [Serializable]
    public class InspectionSettings
    {
        // =================================================================
        // 1. Camera OPS (7 支相機)
        // =================================================================
        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 1 OPS (um)")]
        public double Cam1_Ops { get; set; } = 33.0;

        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 2 OPS (um)")]
        public double Cam2_Ops { get; set; } = 33.0;

        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 3 OPS (um)")]
        public double Cam3_Ops { get; set; } = 33.0;

        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 4 OPS (um)")]
        public double Cam4_Ops { get; set; } = 33.0;

        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 5 OPS (um)")]
        public double Cam5_Ops { get; set; } = 33.0;

        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 6 OPS (um)")]
        public double Cam6_Ops { get; set; } = 33.0;

        [Category("1. 相機解析度 (OPS)")]
        [DisplayName("Cam 7 OPS (um)")]
        public double Cam7_Ops { get; set; } = 33.0;

        // =================================================================
        // 2. Camera Start Position (7 支相機)
        // =================================================================
        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 1 Start (mm)")]
        public double Cam1_Pos { get; set; } = 0.0;

        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 2 Start (mm)")]
        public double Cam2_Pos { get; set; } = 400.0;

        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 3 Start (mm)")]
        public double Cam3_Pos { get; set; } = 800.0;

        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 4 Start (mm)")]
        public double Cam4_Pos { get; set; } = 1200.0;

        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 5 Start (mm)")]
        public double Cam5_Pos { get; set; } = 1600.0;

        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 6 Start (mm)")]
        public double Cam6_Pos { get; set; } = 2000.0;

        [Category("2. 相機起始位置 (Position)")]
        [DisplayName("Cam 7 Start (mm)")]
        public double Cam7_Pos { get; set; } = 2400.0;

        // =================================================================
        // 3. Algorithm Parameters
        // =================================================================
        [Category("3. 演算法參數")]
        [DisplayName("Hessian Max Factor")]
        [Description("控制 Hessian Ridge 的最大值常數 (hessian_max_const)")]
        public float HessianMaxFactor { get; set; } = 5.0f;

        [Category("3. 演算法參數")]
        [DisplayName("Error Value Mean")]
        public float ErrorValueMean { get; set; } = 1.0f;

        [Category("3. 演算法參數")]
        [DisplayName("Error Value Max")]
        public float ErrorValueMax { get; set; } = 2.0f;

        // =================================================================
        // Helper Methods (Array Conversion)
        // =================================================================
        public double[] GetOpsArray()
        {
            return new double[] { Cam1_Ops, Cam2_Ops, Cam3_Ops, Cam4_Ops, Cam5_Ops, Cam6_Ops, Cam7_Ops };
        }

        public double[] GetPosArray()
        {
            return new double[] { Cam1_Pos, Cam2_Pos, Cam3_Pos, Cam4_Pos, Cam5_Pos, Cam6_Pos, Cam7_Pos };
        }

        // =================================================================
        // 儲存與載入邏輯 (利用 XML 序列化存入 UserConfig string)
        // =================================================================
        public static InspectionSettings LoadFromSettings()
        {
            try
            {
                string xml = Properties.Settings.Default.InspectionConfigJson;
                if (string.IsNullOrWhiteSpace(xml)) return new InspectionSettings();

                XmlSerializer serializer = new XmlSerializer(typeof(InspectionSettings));
                using (StringReader reader = new StringReader(xml))
                {
                    return (InspectionSettings)serializer.Deserialize(reader);
                }
            }
            catch
            {
                return new InspectionSettings(); // 載入失敗則回傳預設值
            }
        }

        public void SaveToSettings()
        {
            try
            {
                XmlSerializer serializer = new XmlSerializer(typeof(InspectionSettings));
                using (StringWriter writer = new StringWriter())
                {
                    serializer.Serialize(writer, this);
                    Properties.Settings.Default.InspectionConfigJson = writer.ToString();
                    Properties.Settings.Default.Save();
                }
            }
            catch { /* Ignore save error */ }
        }
    }
}