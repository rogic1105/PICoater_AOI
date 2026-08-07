using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Forms;
using AniloxRoll.Monitor.UI.Widgets;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    public sealed class PropertyGridNumericWheelInterceptorTests
    {
        [TestCase(0.3f, 1, 0.4f)]
        [TestCase(0.3f, 3, 0.6f)]
        [TestCase(0.3f, -4, 0.1f)]
        public void DecimalStep_UsesPointOneAndClampsToPositive(float current, int notches, float expected)
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                current, typeof(float), 0.1m, notches, out object next);

            Assert.That(changed, Is.True);
            Assert.That((float)next, Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void RidgeSigmaStep_UsesWholeNumbers()
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                9f, typeof(float), 1m, 2, out object next);

            Assert.That(changed, Is.True);
            Assert.That((float)next, Is.EqualTo(11f));
        }

        [Test]
        public void ZeroBasedSetting_CanRemainAtZero()
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                0d,
                typeof(double),
                new NumericWheelRule(1m, 0m),
                -1,
                out object next);

            Assert.That(changed, Is.False);
            Assert.That((double)next, Is.EqualTo(0d));
        }

        [Test]
        public void BoundedSetting_ClampsAtMaximum()
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                254,
                typeof(int),
                new NumericWheelRule(1m, 0m, 255m),
                3,
                out object next);

            Assert.That(changed, Is.True);
            Assert.That((int)next, Is.EqualTo(255));
        }

        [Test]
        public void RuleTable_CoversEveryEditableVisibleNumericSetting()
        {
            string[] numericProperties = TypeDescriptor.GetProperties(typeof(InspectionSettings))
                .Cast<PropertyDescriptor>()
                .Where(property => property.IsBrowsable && !property.IsReadOnly)
                .Where(property => IsNumericType(property.PropertyType))
                .Select(property => property.Name)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
            string[] configuredProperties = AniloxRollForm.CreatePropertyGridNumericWheelRules()
                .Keys
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();

            Assert.That(configuredProperties, Is.EqualTo(numericProperties));
        }

        [Test]
        public void RuleTable_UsesJsonBackedStepsIndependently()
        {
            var settings = new PropertyGridWheelSettings
            {
                ColumnNormalizationStep = 0.25m,
                RowNormalizationStep = 0.5m,
                ColumnMeanThresholdStep = 0.02m,
                ColumnMaxThresholdStep = 0.03m
            };

            Dictionary<string, NumericWheelRule> rules =
                AniloxRollForm.CreatePropertyGridNumericWheelRules(settings);

            Assert.That(rules[nameof(InspectionSettings.dc_HessianMaxFactorV)].Step, Is.EqualTo(0.25m));
            Assert.That(rules[nameof(InspectionSettings.dd_HessianMaxFactorH)].Step, Is.EqualTo(0.5m));
            Assert.That(rules[nameof(InspectionSettings.ec_ErrorValueMeanV)].Step, Is.EqualTo(0.02m));
            Assert.That(rules[nameof(InspectionSettings.ed_ErrorValueMaxV)].Step, Is.EqualTo(0.03m));
        }

        [Test]
        public void InvalidJsonBackedStep_FallsBackToDefault()
        {
            var settings = new PropertyGridWheelSettings
            {
                ColumnNormalizationStep = 0m
            };

            NumericWheelRule rule = AniloxRollForm.CreatePropertyGridNumericWheelRules(settings)
                [nameof(InspectionSettings.dc_HessianMaxFactorV)];

            Assert.That(rule.Step, Is.EqualTo(0.1m));
        }

        [Test]
        public void ZeroNotches_DoesNotChangeValue()
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                0.3f, typeof(float), 0.1m, 0, out object next);

            Assert.That(changed, Is.False);
            Assert.That(next, Is.EqualTo(0.3f));
        }

        [Test]
        public void ClickingSameNumericProperty_TogglesWheelEditingOff()
        {
            string armed = PropertyGridNumericWheelInterceptor.ResolveArmedProperty(null, "factor");
            Assert.That(armed, Is.EqualTo("factor"));

            armed = PropertyGridNumericWheelInterceptor.ResolveArmedProperty(armed, "factor");
            Assert.That(armed, Is.Null);
        }

        [Test]
        public void ClickingAnotherProperty_MovesWheelEditingArm()
        {
            string armed = PropertyGridNumericWheelInterceptor.ResolveArmedProperty("mean", "max");

            Assert.That(armed, Is.EqualTo("max"));
        }

        [Test]
        public void TwoClicks_OnlyExposeArmedBlueOrDisarmedUnselectedStates()
        {
            using (var form = new Form())
            using (var grid = new PropertyGrid())
            {
                grid.Dock = DockStyle.Fill;
                grid.ToolbarVisible = false;
                grid.SelectedObject = new NumericSettingsProbe();
                form.Controls.Add(grid);
                form.Show();
                Application.DoEvents();

                GridItem property = FindProperty(grid.SelectedGridItem, nameof(NumericSettingsProbe.Factor));
                Assert.That(property, Is.Not.Null);
                grid.SelectedGridItem = property;
                var armedStates = new List<bool>();
                var interceptor = new PropertyGridNumericWheelInterceptor(
                    grid,
                    new Dictionary<string, NumericWheelRule>
                    {
                        { nameof(NumericSettingsProbe.Factor), new NumericWheelRule(0.1m, 0.1m) }
                    },
                    null,
                    (name, armed) => armedStates.Add(armed));
                Control gridView = FindGridView(grid);
                Assert.That(gridView, Is.Not.Null);

                SendClick(gridView);
                Application.DoEvents();
                Assert.That(grid.ContainsFocus, Is.True);
                Assert.That(armedStates, Is.EqualTo(new[] { true }));

                SendClick(gridView);
                Application.DoEvents();

                Assert.That(grid.ContainsFocus, Is.False);
                Assert.That(armedStates, Is.EqualTo(new[] { true, false }));
                Assert.That(grid.SelectedGridItem?.PropertyDescriptor?.Name,
                    Is.EqualTo(nameof(NumericSettingsProbe.Factor)));

                SendWheel(gridView, 120);
                Application.DoEvents();

                Assert.That(grid.ContainsFocus, Is.False);
                Assert.That(armedStates, Is.EqualTo(new[] { true, false }));
                Assert.That(((NumericSettingsProbe)grid.SelectedObject).Factor, Is.EqualTo(0.3f));
                GC.KeepAlive(interceptor);
            }
        }

        private static void SendClick(Control control)
        {
            IntPtr position = new IntPtr((40 << 16) | 10);
            SendMessage(control.Handle, 0x0201, new IntPtr(1), position);
            SendMessage(control.Handle, 0x0202, IntPtr.Zero, position);
        }

        private static void SendWheel(Control control, int delta)
        {
            IntPtr wParam = new IntPtr((delta & 0xFFFF) << 16);
            SendMessage(control.Handle, 0x020A, wParam, IntPtr.Zero);
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

        private static GridItem FindProperty(GridItem item, string propertyName)
        {
            if (item == null)
                return null;
            if (string.Equals(item.PropertyDescriptor?.Name, propertyName, StringComparison.Ordinal))
                return item;

            foreach (GridItem child in item.GridItems)
            {
                GridItem match = FindProperty(child, propertyName);
                if (match != null)
                    return match;
            }
            return null;
        }

        private static bool IsNumericType(Type type)
        {
            Type actualType = Nullable.GetUnderlyingType(type) ?? type;
            if (actualType.IsEnum)
                return false;
            switch (Type.GetTypeCode(actualType))
            {
                case TypeCode.Byte:
                case TypeCode.SByte:
                case TypeCode.Int16:
                case TypeCode.UInt16:
                case TypeCode.Int32:
                case TypeCode.UInt32:
                case TypeCode.Int64:
                case TypeCode.UInt64:
                case TypeCode.Single:
                case TypeCode.Double:
                case TypeCode.Decimal:
                    return true;
                default:
                    return false;
            }
        }

        private sealed class NumericSettingsProbe
        {
            public float Factor { get; set; } = 0.3f;
        }

        [DllImport("user32.dll")]
        private static extern IntPtr SendMessage(
            IntPtr windowHandle,
            int message,
            IntPtr wParam,
            IntPtr lParam);
    }
}
