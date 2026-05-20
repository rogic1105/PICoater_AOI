using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.Core.Data
{
    public class DcfFileEditor : UITypeEditor
    {
        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context)
            => UITypeEditorEditStyle.Modal;

        public override object EditValue(
            ITypeDescriptorContext context, IServiceProvider provider, object value)
        {
            using (var dlg = new OpenFileDialog())
            {
                dlg.Filter = "DCF 設定檔 (*.dcf)|*.dcf|所有檔案 (*.*)|*.*";
                dlg.FilterIndex = 1;
                dlg.Title = "選擇 DCF 設定檔";
                if (value is string s && !string.IsNullOrEmpty(s))
                    dlg.FileName = s;
                if (dlg.ShowDialog() == DialogResult.OK)
                    return dlg.FileName;
            }
            return value;
        }
    }
}
