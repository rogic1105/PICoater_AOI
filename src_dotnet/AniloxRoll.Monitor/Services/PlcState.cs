namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>PLC FSM 狀態。</summary>
    public enum PlcState
    {
        Disconnected,   // 未連線
        Idle,           // 待機（原 Ready）
        Running,        // 取像中（原 Run）
        Stopping,       // 停止中（原 Stop）
        Faulted,        // PLC 故障（原 PLCErr）
        CommLost,       // 通訊中斷（原 PCErr）
        Closed          // 已關閉
    }

    /// <summary>PLC IO 快照（每次 PollTick 結束時發布）。</summary>
    public struct PlcIoSnapshot
    {
        public bool DiPlcAlive;
        public bool DiStart;
        public bool DoPcAlive;
        public bool DoMura;
        public bool DoPcBusy;
    }
}
