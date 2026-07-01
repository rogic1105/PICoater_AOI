using System;
using System.Windows.Forms;

namespace TanukiCv.Controls.WinForms
{
    public static class PropertyGridScrollKeeper
    {
        public static bool RefreshGridItem(PropertyGrid grid, string propertyName, Action<bool> setSuppressSelectionChanged = null)
        {
            if (grid == null || string.IsNullOrEmpty(propertyName)) return false;

            GridItem root = grid.SelectedGridItem;
            while (root?.Parent != null) root = root.Parent;
            if (root == null) return false;

            GridItem found = FindGridItemRecursive(root, propertyName);
            if (found == null) return false;

            var scroll = CaptureScrollBar(grid);
            var saved = grid.SelectedGridItem;

            setSuppressSelectionChanged?.Invoke(true);
            try
            {
                grid.SelectedGridItem = found;
                if (saved != null && saved != found)
                    grid.SelectedGridItem = saved;
                scroll.Restore();
            }
            finally
            {
                setSuppressSelectionChanged?.Invoke(false);
            }

            return true;
        }

        public static void RefreshKeepScroll(PropertyGrid grid)
        {
            if (grid == null) return;

            Control gridView = FindPropertyGridView(grid);
            if (gridView == null)
            {
                grid.Refresh();
                return;
            }

            var scroll = CaptureScrollBar(grid);
            using (new RedrawScope(grid, gridView))
            {
                grid.Refresh();
                scroll.Restore();
            }
        }

        private static GridItem FindGridItemRecursive(GridItem parent, string name)
        {
            if (parent == null) return null;
            foreach (GridItem child in parent.GridItems)
            {
                if (child.PropertyDescriptor?.Name == name) return child;
                var nested = FindGridItemRecursive(child, name);
                if (nested != null) return nested;
            }
            return null;
        }

        private static ScrollSnapshot CaptureScrollBar(PropertyGrid grid)
        {
            Control gridView = FindPropertyGridView(grid);
            if (gridView == null) return new ScrollSnapshot(null, 0);

            foreach (Control child in gridView.Controls)
            {
                if (child is VScrollBar scrollBar)
                    return new ScrollSnapshot(scrollBar, scrollBar.Value);
            }

            return new ScrollSnapshot(null, 0);
        }

        private static Control FindPropertyGridView(PropertyGrid grid)
        {
            if (grid == null) return null;
            foreach (Control child in grid.Controls)
            {
                if (child.GetType().Name == "PropertyGridView")
                    return child;
            }
            return null;
        }

        private struct ScrollSnapshot
        {
            private readonly ScrollBar _scrollBar;
            private readonly int _value;

            public ScrollSnapshot(ScrollBar scrollBar, int value)
            {
                _scrollBar = scrollBar;
                _value = value;
            }

            public void Restore()
            {
                if (_scrollBar == null) return;

                int max = Math.Max(0, _scrollBar.Maximum - _scrollBar.LargeChange + 1);
                _scrollBar.Value = Math.Max(0, Math.Min(_value, max));
            }
        }
    }
}
