using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Product capture policy for IO-originated stop requests.
    ///
    /// State + Event -> Next + Action:
    /// IO capture + StartLow -> TailDrain + stop after one complete tail frame.
    /// IO capture + PlcAliveLost/CommunicationLost -> Stop + do not wait for tail.
    /// Time/Height capture + any IO stop request -> Capturing + ignore product stop.
    /// </summary>
    internal static class CaptureStopPolicy
    {
        public static bool ShouldStopOnIoRequest(CaptureStopCondition condition)
        {
            return condition == CaptureStopCondition.IoSignal;
        }

        public static bool ShouldDrainIoTail(IoStopRequestReason reason)
        {
            return reason == IoStopRequestReason.StartLow;
        }
    }
}
