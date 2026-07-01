using System;
using System.Windows.Forms;

namespace TanukiCv.Controls.WinForms
{
    public sealed class ListViewScrollKeeper
    {
        private readonly string _topItemText;
        private readonly int _topIndex;

        private ListViewScrollKeeper(string topItemText, int topIndex)
        {
            _topItemText = topItemText;
            _topIndex = topIndex;
        }

        public static ListViewScrollKeeper Capture(ListView listView)
        {
            if (listView == null || listView.Items.Count == 0)
                return new ListViewScrollKeeper(null, 0);

            try
            {
                var topItem = listView.TopItem;
                return topItem == null
                    ? new ListViewScrollKeeper(null, 0)
                    : new ListViewScrollKeeper(topItem.Text, topItem.Index);
            }
            catch (InvalidOperationException)
            {
                return new ListViewScrollKeeper(null, 0);
            }
        }

        public void Restore(ListView listView)
        {
            if (listView == null || listView.Items.Count == 0) return;

            int restoreIndex = FindRestoreIndex(listView);
            try
            {
                listView.TopItem = listView.Items[restoreIndex];
            }
            catch (InvalidOperationException)
            {
                listView.BeginInvoke(new Action(() =>
                {
                    if (!listView.IsDisposed && listView.Items.Count > restoreIndex)
                        listView.TopItem = listView.Items[restoreIndex];
                }));
            }
        }

        private int FindRestoreIndex(ListView listView)
        {
            if (!string.IsNullOrEmpty(_topItemText))
            {
                for (int i = 0; i < listView.Items.Count; i++)
                {
                    if (listView.Items[i].Text == _topItemText)
                        return i;
                }
            }

            return Math.Max(0, Math.Min(_topIndex, listView.Items.Count - 1));
        }
    }
}
