namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 可重入的 bool 旗標，搭配 <see cref="EventGuardScope"/> 以 using 語法自動還原。
    /// 取代散落各處的 try { flag=true; ... } finally { flag=false; } 模式。
    /// </summary>
    public sealed class EventGuard
    {
        private int _depth;

        /// <summary>旗標是否被持有（depth &gt; 0）。</summary>
        public bool IsSet => _depth > 0;

        /// <summary>取得一個 scope；離開 using 區塊時自動釋放。</summary>
        public EventGuardScope Enter()
        {
            _depth++;
            return new EventGuardScope(this);
        }

        internal void Exit()
        {
            if (_depth > 0) _depth--;
        }
    }

    /// <summary>
    /// <see cref="EventGuard.Enter"/> 回傳的輕量 struct，離開 using 區塊時自動呼叫 Exit。
    /// C# 7.3 不支援 ref struct 的 IDisposable，故以一般 struct 實作。
    /// </summary>
    public struct EventGuardScope : System.IDisposable
    {
        private EventGuard _owner;

        internal EventGuardScope(EventGuard owner) => _owner = owner;

        public void Dispose()
        {
            _owner?.Exit();
            _owner = null;
        }
    }
}
