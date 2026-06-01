using System;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// PropertyGrid 同一 Category 內的顯示順序（數字小排前）。
    /// 沒標 = order 0（同 order 維持原 declaration 順序）。
    /// 由 InspectionSettingsDescriptionProvider 解析；沒這個 Provider 的物件不生效。
    /// </summary>
    [AttributeUsage(AttributeTargets.Property)]
    internal sealed class PropertyOrderAttribute : Attribute
    {
        public int Order { get; }
        public PropertyOrderAttribute(int order) { Order = order; }
    }
}
