using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// Changes explicitly opted-in numeric PropertyGrid values with the mouse wheel.
    /// Non-opted-in rows retain the PropertyGrid's normal scrolling behavior.
    /// </summary>
    internal sealed class PropertyGridNumericWheelInterceptor : NativeWindow
    {
        private const int WmMouseWheel = 0x020A;
        private const int WmLeftButtonDown = 0x0201;
        private const int WheelDelta = 120;

        private readonly PropertyGrid _grid;
        private readonly IDictionary<string, decimal> _steps;
        private readonly Action<string, object, object> _valueChanged;
        private readonly Action<string, bool> _armedChanged;
        private Control _gridView;
        private int _wheelRemainder;
        private string _armedPropertyName;

        public PropertyGridNumericWheelInterceptor(
            PropertyGrid grid,
            IDictionary<string, decimal> steps,
            Action<string, object, object> valueChanged,
            Action<string, bool> armedChanged = null)
        {
            _grid = grid ?? throw new ArgumentNullException(nameof(grid));
            _steps = steps ?? throw new ArgumentNullException(nameof(steps));
            _valueChanged = valueChanged;
            _armedChanged = armedChanged;

            _grid.HandleCreated += OnGridHandleCreated;
            _grid.SelectedGridItemChanged += OnSelectedGridItemChanged;
            AttachToGridView();
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WmLeftButtonDown)
            {
                base.WndProc(ref m);
                ToggleSelectedProperty();
                return;
            }

            if (m.Msg == WmMouseWheel && TryApplyWheelDelta(GetWheelDelta(m.WParam)))
                return;

            base.WndProc(ref m);
        }

        private bool TryApplyWheelDelta(int delta)
        {
            GridItem item = _grid.SelectedGridItem;
            PropertyDescriptor descriptor = item?.PropertyDescriptor;
            if (descriptor == null || descriptor.IsReadOnly ||
                !string.Equals(_armedPropertyName, descriptor.Name, StringComparison.Ordinal) ||
                !_steps.TryGetValue(descriptor.Name, out decimal step))
            {
                _wheelRemainder = 0;
                return false;
            }

            _wheelRemainder += delta;
            int notches = _wheelRemainder / WheelDelta;
            if (notches == 0)
                return true;
            _wheelRemainder -= notches * WheelDelta;

            object component = _grid.SelectedObject;
            object oldValue = descriptor.GetValue(component);
            if (!TryCalculateNext(oldValue, descriptor.PropertyType, step, notches, out object newValue))
                return true;

            try
            {
                descriptor.SetValue(component, newValue);
                _grid.Refresh();
                _valueChanged?.Invoke(descriptor.Name, oldValue, newValue);
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[PropertyGridNumericWheelInterceptor] {ex.GetType().Name}: {ex.Message}");
            }
            return true;
        }

        private void ToggleSelectedProperty()
        {
            PropertyDescriptor descriptor = _grid.SelectedGridItem?.PropertyDescriptor;
            string clickedName = descriptor != null && !descriptor.IsReadOnly &&
                _steps.ContainsKey(descriptor.Name)
                ? descriptor.Name
                : null;
            string next = ResolveArmedProperty(_armedPropertyName, clickedName);
            if (string.Equals(next, _armedPropertyName, StringComparison.Ordinal))
                return;

            string previous = _armedPropertyName;
            _armedPropertyName = next;
            _wheelRemainder = 0;
            if (!string.IsNullOrEmpty(previous))
                _armedChanged?.Invoke(previous, false);
            if (!string.IsNullOrEmpty(next))
                _armedChanged?.Invoke(next, true);
        }

        internal static string ResolveArmedProperty(string armedPropertyName, string clickedPropertyName)
        {
            if (string.IsNullOrEmpty(clickedPropertyName))
                return null;
            return string.Equals(armedPropertyName, clickedPropertyName, StringComparison.Ordinal)
                ? null
                : clickedPropertyName;
        }

        internal static bool TryCalculateNext(
            object currentValue,
            Type propertyType,
            decimal step,
            int notches,
            out object nextValue)
        {
            nextValue = currentValue;
            if (currentValue == null || step <= 0 || notches == 0)
                return false;

            try
            {
                decimal current = Convert.ToDecimal(currentValue, CultureInfo.InvariantCulture);
                decimal next = current + step * notches;
                if (next < step)
                    next = step;
                next = decimal.Round(next, DecimalPlaces(step), MidpointRounding.AwayFromZero);

                if (propertyType == typeof(float)) nextValue = (float)next;
                else if (propertyType == typeof(double)) nextValue = (double)next;
                else if (propertyType == typeof(decimal)) nextValue = next;
                else if (propertyType == typeof(int)) nextValue = decimal.ToInt32(next);
                else return false;

                return !Equals(currentValue, nextValue);
            }
            catch (Exception ex) when (
                ex is FormatException ||
                ex is InvalidCastException ||
                ex is OverflowException)
            {
                return false;
            }
        }

        private static int DecimalPlaces(decimal value)
        {
            int[] bits = decimal.GetBits(value);
            return (bits[3] >> 16) & 0x7F;
        }

        private static int GetWheelDelta(IntPtr wParam)
        {
            return (short)(((long)wParam >> 16) & 0xFFFF);
        }

        private void OnGridHandleCreated(object sender, EventArgs e)
        {
            AttachToGridView();
        }

        private void OnSelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            _wheelRemainder = 0;
            PropertyDescriptor descriptor = e.NewSelection?.PropertyDescriptor;
            if (_armedPropertyName != null &&
                !string.Equals(_armedPropertyName, descriptor?.Name, StringComparison.Ordinal))
            {
                string previous = _armedPropertyName;
                _armedPropertyName = null;
                _armedChanged?.Invoke(previous, false);
            }
        }

        private void AttachToGridView()
        {
            Control next = FindGridView(_grid);
            if (next == null || ReferenceEquals(next, _gridView))
                return;

            if (_gridView != null)
            {
                _gridView.HandleCreated -= OnGridViewHandleCreated;
                _gridView.HandleDestroyed -= OnGridViewHandleDestroyed;
            }
            if (Handle != IntPtr.Zero)
                ReleaseHandle();

            _gridView = next;
            _gridView.HandleCreated += OnGridViewHandleCreated;
            _gridView.HandleDestroyed += OnGridViewHandleDestroyed;
            if (_gridView.IsHandleCreated)
                AssignHandle(_gridView.Handle);
        }

        private void OnGridViewHandleCreated(object sender, EventArgs e)
        {
            if (Handle != IntPtr.Zero)
                ReleaseHandle();
            AssignHandle(_gridView.Handle);
        }

        private void OnGridViewHandleDestroyed(object sender, EventArgs e)
        {
            if (Handle != IntPtr.Zero)
                ReleaseHandle();
        }

        private static Control FindGridView(Control root)
        {
            foreach (Control child in root.Controls)
            {
                if (child.GetType().Name == "PropertyGridView")
                    return child;
                Control nested = FindGridView(child);
                if (nested != null)
                    return nested;
            }
            return null;
        }
    }
}
