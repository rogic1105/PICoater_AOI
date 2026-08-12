using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using TanukiCv.Controls.WinForms;

namespace AniloxRoll.Monitor.UI.Binders
{
    public sealed class GrabDetailRowCommittedEventArgs : EventArgs
    {
        public GrabDetailRowCommittedEventArgs(string grabId, bool isRepeated)
        {
            GrabId = grabId;
            IsRepeated = isRepeated;
        }

        public string GrabId { get; }
        public bool IsRepeated { get; }
    }

    /// <summary>
    /// Owns the report detail ListView wiring, virtual rows, drawing, scrolling and selection visuals.
    /// Report selection policy remains in DataStatisticsPresenter.
    /// </summary>
    public sealed class GrabDetailListBinder : IDisposable
    {
        private static readonly Color DetailPass = Color.FromArgb(232, 245, 233);
        private static readonly Color DetailFail = Color.FromArgb(255, 235, 238);
        private static readonly Color DetailUnknown = SystemColors.Window;

        private readonly ListView _listView;
        private readonly int _cameraCount;
        private List<GrabDetail> _visibleDetails = new List<GrabDetail>();
        private string _selectedGrabId;
        private bool _initialized;
        private long _lastVirtualFallbackTicks;

        public GrabDetailListBinder(ListView listView, int cameraCount)
        {
            _listView = listView ?? throw new ArgumentNullException(nameof(listView));
            _cameraCount = cameraCount;
        }

        public Func<bool> IsSelectionActive { private get; set; }
        public int VisibleCount => _listView.VirtualListSize;
        public event EventHandler<GrabDetailRowCommittedEventArgs> RowCommitted;

        public void Initialize()
        {
            if (_initialized) return;
            _initialized = true;

            _listView.View = View.Details;
            _listView.FullRowSelect = true;
            _listView.GridLines = true;
            _listView.OwnerDraw = true;
            _listView.VirtualMode = true;
            EnableDoubleBuffering(_listView);
            _listView.Columns.Clear();
            _listView.VirtualListSize = 0;

            _listView.Columns.Add("序號", -1, HorizontalAlignment.Center);
            for (int i = 1; i <= _cameraCount; i++)
                _listView.Columns.Add($"{i}", -1, HorizontalAlignment.Center);
            _listView.Columns.Add("列", -1, HorizontalAlignment.Center);
            FitColumnsToContent();

            _listView.DrawColumnHeader += OnDrawColumnHeader;
            _listView.DrawSubItem += OnDrawSubItem;
            _listView.RetrieveVirtualItem += OnRetrieveVirtualItem;
            _listView.MouseUp += OnMouseUp;
            _listView.Resize += OnResize;
        }

        public void SetItems(List<GrabDetail> details)
        {
            _listView.BeginUpdate();
            try
            {
                _listView.SelectedIndices.Clear();
                // Keep the native virtual-list size and the managed snapshot in lockstep.
                // Windows may request an old row while VirtualListSize is changing, so the
                // RetrieveVirtualItem handler must stay attached throughout this transition.
                _listView.VirtualListSize = 0;
                _visibleDetails = details == null
                    ? new List<GrabDetail>()
                    : new List<GrabDetail>(details);
                _listView.VirtualListSize = _visibleDetails.Count;
                FitColumnsToContent();
            }
            finally
            {
                _listView.EndUpdate();
            }
        }

        public void Highlight(string grabId)
        {
            int previousRow = FindRow(_selectedGrabId);
            _selectedGrabId = grabId;
            int row = FindRow(grabId);
            if (row >= 0 && _listView.IsHandleCreated && row < _listView.VirtualListSize)
                EnsureRowInBufferedViewport(row);
            RedrawRow(previousRow);
            RedrawRow(row);
        }

        public void Refresh(string grabId)
        {
            RedrawRow(FindRow(grabId));
        }

        public void RefreshAll()
        {
            if (_listView.IsHandleCreated && !_listView.IsDisposed)
                _listView.Invalidate();
        }

        public void ClearSelection()
        {
            int previousRow = FindRow(_selectedGrabId);
            _selectedGrabId = null;
            _listView.SelectedIndices.Clear();
            RedrawRow(previousRow);
        }

        public void ExecuteWithRedrawSuspended(Action action)
        {
            if (action == null) return;
            if (!_listView.IsHandleCreated || _listView.IsDisposed)
            {
                action();
                return;
            }

            using (new RedrawScope(_listView))
                action();
        }

        public void Dispose()
        {
            if (!_initialized) return;
            // Drain native virtual-row requests before detaching RetrieveVirtualItem. Leaving a
            // non-zero VirtualListSize without a handler causes an endless WinForms exception
            // loop during teardown or a late redraw.
            if (!_listView.IsDisposed)
            {
                _listView.BeginUpdate();
                try
                {
                    _listView.SelectedIndices.Clear();
                    _listView.VirtualListSize = 0;
                    _visibleDetails = new List<GrabDetail>();
                }
                finally
                {
                    _listView.EndUpdate();
                }
            }
            _listView.DrawColumnHeader -= OnDrawColumnHeader;
            _listView.DrawSubItem -= OnDrawSubItem;
            _listView.RetrieveVirtualItem -= OnRetrieveVirtualItem;
            _listView.MouseUp -= OnMouseUp;
            _listView.Resize -= OnResize;
            _initialized = false;
        }

        private void OnResize(object sender, EventArgs e)
        {
            FitColumnsToContent();
        }

        private void OnMouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left || _listView.SelectedIndices.Count == 0) return;
            int row = _listView.SelectedIndices[0];
            if (row < 0 || row >= _visibleDetails.Count) return;

            string grabId = _visibleDetails[row].GrabId;
            if (string.IsNullOrEmpty(grabId)) return;
            bool repeated = grabId == _selectedGrabId;
            _selectedGrabId = grabId;
            RowCommitted?.Invoke(this, new GrabDetailRowCommittedEventArgs(grabId, repeated));
        }

        private int FindRow(string grabId) =>
            string.IsNullOrEmpty(grabId) ? -1 : _visibleDetails.FindIndex(d => d.GrabId == grabId);

        private void OnRetrieveVirtualItem(object sender, RetrieveVirtualItemEventArgs e)
        {
            // Assign a valid item first. Even malformed/stale data must never escape this event
            // without e.Item, because WinForms turns that into a repeating UI-thread exception.
            e.Item = BuildPlaceholderItem();
            try
            {
                if (e.ItemIndex < 0 || e.ItemIndex >= _visibleDetails.Count)
                {
                    LogVirtualFallback(e.ItemIndex, "stale-index");
                    return;
                }
                e.Item = BuildItem(e.ItemIndex);
            }
            catch (Exception ex)
            {
                LogVirtualFallback(e.ItemIndex, ex.GetType().Name);
            }
        }

        private ListViewItem BuildItem(int index)
        {
            if (index < 0 || index >= _visibleDetails.Count)
                return BuildPlaceholderItem();

            var detail = _visibleDetails[index];
            if (detail == null)
                return BuildPlaceholderItem();
            var item = new ListViewItem(detail.GrabId)
            {
                UseItemStyleForSubItems = false,
                BackColor = DetailUnknown
            };
            bool rowHasFail = false;
            for (int i = 0; i < _cameraCount; i++)
            {
                bool? result = detail.CamResult != null && i < detail.CamResult.Length
                    ? detail.CamResult[i]
                    : null;
                AddResultSubItem(item, result);
                rowHasFail |= result == true;
            }
            AddResultSubItem(item, detail.RowResult);
            rowHasFail |= detail.RowResult == true;

            item.Tag = rowHasFail;
            return item;
        }

        private static void AddResultSubItem(ListViewItem item, bool? failed)
        {
            string text = !failed.HasValue ? "—" : failed.Value ? "×" : "○";
            ListViewItem.ListViewSubItem subItem = item.SubItems.Add(text);
            subItem.BackColor = GetResultBackColor(failed);
        }

        internal static Color GetResultBackColor(bool? failed)
        {
            if (!failed.HasValue) return DetailUnknown;
            return failed.Value ? DetailFail : DetailPass;
        }

        private ListViewItem BuildPlaceholderItem()
        {
            var item = new ListViewItem(string.Empty)
            {
                Tag = false,
                BackColor = DetailUnknown,
                UseItemStyleForSubItems = false
            };
            for (int i = 0; i <= _cameraCount; i++)
                AddResultSubItem(item, null);
            return item;
        }

        private void LogVirtualFallback(int index, string reason)
        {
            long now = Stopwatch.GetTimestamp();
            long previous = _lastVirtualFallbackTicks;
            if (previous != 0 && now - previous < Stopwatch.Frequency * 5L)
                return;
            _lastVirtualFallbackTicks = now;
            FlowTrace.Log(
                $"DT list virtual fallback index={index} rows={_visibleDetails.Count} " +
                $"native={_listView.VirtualListSize} reason={reason}");
        }

        private static void OnDrawColumnHeader(object sender, DrawListViewColumnHeaderEventArgs e)
        {
            e.DrawDefault = true;
        }

        private void OnDrawSubItem(object sender, DrawListViewSubItemEventArgs e)
        {
            bool rowHasFail = e.Item.Tag is bool failed && failed;
            Color backColor = e.SubItem.BackColor;
            using (var brush = new SolidBrush(backColor))
                e.Graphics.FillRectangle(brush, e.Bounds);

            var flags = TextFormatFlags.VerticalCenter
                      | TextFormatFlags.HorizontalCenter
                      | TextFormatFlags.EndEllipsis
                      | TextFormatFlags.NoPrefix;
            TextRenderer.DrawText(e.Graphics, e.SubItem.Text, e.Item.ListView.Font,
                e.Bounds, e.Item.ForeColor, flags);

            bool active = IsSelectionActive?.Invoke() == true;
            if (!active || e.Item.Text != _selectedGrabId || e.ColumnIndex != e.Item.SubItems.Count - 1)
                return;

            Rectangle bounds = e.Item.Bounds;
            bounds.Width = e.Item.ListView.Columns.Cast<ColumnHeader>().Sum(c => c.Width);
            bounds.Width = Math.Min(bounds.Width, e.Item.ListView.ClientSize.Width - 1);
            bounds.Height -= 1;
            if (bounds.Width <= 0 || bounds.Height <= 0) return;

            Color border = rowHasFail ? Color.FromArgb(211, 47, 47) : Color.FromArgb(46, 125, 50);
            using (var pen = new Pen(border, 2))
                e.Graphics.DrawRectangle(pen, bounds);
        }

        private void FitColumnsToContent()
        {
            if (_listView.Columns.Count == 0) return;
            using (var graphics = _listView.CreateGraphics())
            {
                const int padding = 16;
                string sample = _visibleDetails.Count > 0 &&
                                _visibleDetails[0] != null &&
                                !string.IsNullOrEmpty(_visibleDetails[0].GrabId)
                    ? _visibleDetails[0].GrabId
                    : _listView.Columns[0].Text;
                _listView.Columns[0].Width = (int)Math.Ceiling(Math.Max(
                    graphics.MeasureString(_listView.Columns[0].Text, _listView.Font).Width,
                    graphics.MeasureString(sample, _listView.Font).Width)) + padding;

                float glyphWidth = graphics.MeasureString("×", _listView.Font).Width;
                for (int i = 1; i < _listView.Columns.Count; i++)
                    _listView.Columns[i].Width = (int)Math.Ceiling(Math.Max(
                        graphics.MeasureString(_listView.Columns[i].Text, _listView.Font).Width,
                        glyphWidth)) + padding;
            }
            int itemHeight = Math.Max(18, _listView.Font.Height + 6);
            int visibleRows = Math.Max(1, (_listView.ClientSize.Height - 24) / itemHeight);
            int scrollbarWidth = _visibleDetails.Count > visibleRows
                ? SystemInformation.VerticalScrollBarWidth
                : 0;
            int available = Math.Max(0, _listView.ClientSize.Width - scrollbarWidth - 2);
            int used = _listView.Columns.Cast<ColumnHeader>().Sum(column => column.Width);
            if (available > used)
                _listView.Columns[0].Width += available - used;
        }

        private void EnsureRowInBufferedViewport(int row)
        {
            int itemHeight = Math.Max(18, _listView.Font.Height + 6);
            int visibleRows = Math.Max(1, (_listView.ClientSize.Height - 24) / itemHeight);
            int margin = Math.Min(5, Math.Max(1, visibleRows / 4));
            int top = 0;
            try
            {
                if (_listView.TopItem != null)
                    top = _listView.TopItem.Index;
            }
            catch (InvalidOperationException) { }

            int bottom = Math.Min(_listView.VirtualListSize - 1, top + visibleRows - 1);
            if (row < top + margin)
                _listView.EnsureVisible(Math.Max(0, row - margin));
            else if (row > bottom - margin)
                _listView.EnsureVisible(Math.Min(_listView.VirtualListSize - 1, row + margin));
        }

        private void RedrawRow(int row)
        {
            if (!_listView.IsHandleCreated || row < 0 || row >= _listView.VirtualListSize) return;
            try { _listView.RedrawItems(row, row, true); }
            catch (InvalidOperationException)
            {
                try { _listView.Invalidate(_listView.GetItemRect(row)); }
                catch (InvalidOperationException) { }
            }
            catch (ArgumentOutOfRangeException) { }
        }

        private static void EnableDoubleBuffering(ListView listView)
        {
            try
            {
                typeof(Control).GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?.SetValue(listView, true, null);
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[DataListDoubleBuffer] {ex.GetType().Name}: {ex.Message}");
            }
        }
    }
}
