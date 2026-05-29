using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Settings.Services
{
    /// <summary>
    /// 檢測設定的 Single Source of Truth：所有 setting 變更走這個 Hub，
    /// 統一觸發 [write memory → save disk → raise event] 三段管線。
    ///
    /// 入口三條：
    ///   Set&lt;V&gt;(selector, value)         — 程式碼路徑（chart click、Switch* helpers、AutoDetect 回寫）
    ///   SetBatch(mutator, ...names)         — 多個 setting 一次 save、依序 raise event
    ///   NotifyExternalChange(name, old, new)— PropertyGrid 改值（setter 已寫 memory，Hub 只負責 save + event）
    ///
    /// 訂閱方：Form.OnSettingChanged 用 switch case 接管 Apply* 副作用。
    /// </summary>
    public sealed class SettingsHub
    {
        public InspectionSettings Settings { get; }

        /// <summary>單一變更通知管道。所有 setting 變更都會走過這條 event。</summary>
        public event Action<SettingChange> Changed;

        public SettingsHub(InspectionSettings settings)
        {
            Settings = settings ?? throw new ArgumentNullException(nameof(settings));
        }

        public void Set<V>(Expression<Func<InspectionSettings, V>> selector, V value)
        {
            var prop = ExtractProperty(selector);
            var oldVal = prop.GetValue(Settings);
            if (Equals(oldVal, value)) return;
            prop.SetValue(Settings, value);
            ConfigManager.SaveInspectionSettings(Settings);
            Changed?.Invoke(new SettingChange(prop.Name, oldVal, value, SettingSource.Programmatic));
        }

        /// <summary>
        /// Runtime 設值（給自製 SettingsPanel 內 row 編輯用）。Source = PropertyGrid（view-edit 風格，
        /// OnSettingChanged 內接收方判斷自己已 paint 不重複 refresh）。
        /// </summary>
        public void SetByName(string name, object value)
        {
            var prop = typeof(InspectionSettings).GetProperty(name);
            if (prop == null) throw new ArgumentException($"unknown setting: {name}", nameof(name));
            var oldVal = prop.GetValue(Settings);
            if (Equals(oldVal, value)) return;
            prop.SetValue(Settings, value);
            ConfigManager.SaveInspectionSettings(Settings);
            Changed?.Invoke(new SettingChange(name, oldVal, value, SettingSource.PropertyGrid));
        }

        /// <summary>
        /// 多 setting 一次 save、**不 raise event**。用在「caller 需要嚴格 transition 順序」場景
        /// （例 chart click 切 mode + 關 enhance：caller 自己 await reload + fit，避免 event handler race）。
        /// 想要 event-driven 場景請用 Set / NotifyExternalChange。
        /// </summary>
        public void SetBatch(Action<InspectionSettings> mutator)
        {
            if (mutator == null) throw new ArgumentNullException(nameof(mutator));
            mutator(Settings);
            ConfigManager.SaveInspectionSettings(Settings);
        }

        public void NotifyExternalChange(string name, object oldVal, object newVal)
        {
            ConfigManager.SaveInspectionSettings(Settings);
            Changed?.Invoke(new SettingChange(name, oldVal, newVal, SettingSource.PropertyGrid));
        }

        private static PropertyInfo ExtractProperty<V>(Expression<Func<InspectionSettings, V>> selector)
        {
            if (selector.Body is MemberExpression me && me.Member is PropertyInfo prop) return prop;
            if (selector.Body is UnaryExpression ue && ue.Operand is MemberExpression me2 && me2.Member is PropertyInfo prop2) return prop2;
            throw new ArgumentException("selector must reference a property", nameof(selector));
        }
    }

    public enum SettingSource
    {
        /// <summary>使用者在 PropertyGrid 改值（setter 已寫 memory、UI 已自我更新）</summary>
        PropertyGrid,
        /// <summary>程式碼路徑改值（chart click、AutoDetect 回寫、btnReviewSelectFolder reset 等）— PropertyGrid 顯示需外部刷新</summary>
        Programmatic,
    }

    public sealed class SettingChange
    {
        public string Name { get; }
        public object OldValue { get; }
        public object NewValue { get; }
        public SettingSource Source { get; }

        public SettingChange(string name, object oldVal, object newVal, SettingSource source = SettingSource.Programmatic)
        {
            Name = name; OldValue = oldVal; NewValue = newVal; Source = source;
        }
    }
}
