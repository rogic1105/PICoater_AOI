namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>IO FSM 狀態（ET-7044 ↔ Nakan 設備）。</summary>
    public enum IoState
    {
        Disconnected,   // 未連線
        Idle,           // 待機
        Running,        // 取像中
        AwaitingStartLow, // 固定時間/高度完成，等待 START Low 復歸
        Stopping,       // 停止中
        Faulted,        // 設備離線（DI_NAKAN_ALIVE 消失）
        CommLost,       // 通訊中斷（TCP 例外）
        Closed          // 已關閉
    }

    /// <summary>IO 快照（每次 PollTick 結束時發布）。</summary>
    public struct IoSnapshot
    {
        public bool DiNakanAlive;
        public bool DiInspectStart;
        public bool DoPcAlive;
        public bool DoMuraDetected;
        public bool DoPcInspect;
    }
}
