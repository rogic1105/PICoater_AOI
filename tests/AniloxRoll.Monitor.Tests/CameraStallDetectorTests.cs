using NUnit.Framework;
using AniloxRoll.Monitor.UI.Managers;

namespace AniloxRoll.Monitor.Tests
{
    /// <summary>
    /// CameraStallDetector 單元測試 —— 展示「真提取」的回報：stall 偵測從 LiveCameraManager 提取成
    /// 純邏輯小組後，可**不開相機、不碰 MIL/UI**，純餵幀數序列就能驗判定。
    /// 判據＝M_PROCESS_FRAME_COUNT 有沒有前進；偵測窗＝4 + ceil(幀週期ms×1.5 / 500ms)。
    /// </summary>
    [TestFixture]
    public class CameraStallDetectorTests
    {
        // 幀數一直前進 = grab 活著 → 永遠不該判 stall。
        [Test]
        public void AdvancingFrameCount_NeverStalls()
        {
            var d = new CameraStallDetector();
            for (long fc = 100; fc < 130; fc++)
                Assert.That(d.Update(1, fc, 0), Is.False, $"fc={fc} 前進中不該 stall");
        }

        // 幀數凍住 + 預期 FPS 未知（窗=4）→ 首次 priming 後，第 4 個凍住 tick 判 stall。
        [Test]
        public void FrozenFrameCount_StallsAtWindowEnd()
        {
            var d = new CameraStallDetector();
            Assert.That(d.Update(1, 100, 0), Is.False, "首次（lastFc=-1）視為活著");
            Assert.That(d.Update(1, 100, 0), Is.False, "凍 tick 1");
            Assert.That(d.Update(1, 100, 0), Is.False, "凍 tick 2");
            Assert.That(d.Update(1, 100, 0), Is.False, "凍 tick 3");
            Assert.That(d.Update(1, 100, 0), Is.True,  "凍 tick 4 = 滿窗 → stall");
        }

        // 凍住一半又前進 → 重置；要重新累計滿窗才會 stall（不會殘留）。
        [Test]
        public void FreezeThenAdvance_ResetsAccumulation()
        {
            var d = new CameraStallDetector();
            d.Update(1, 100, 0);                                   // priming
            d.Update(1, 100, 0);                                   // 凍 1
            d.Update(1, 100, 0);                                   // 凍 2
            Assert.That(d.Update(1, 101, 0), Is.False, "幀數前進 → 重置");
            Assert.That(d.Update(1, 101, 0), Is.False, "重置後凍 1");
            Assert.That(d.Update(1, 101, 0), Is.False, "凍 2");
            Assert.That(d.Update(1, 101, 0), Is.False, "凍 3");
            Assert.That(d.Update(1, 101, 0), Is.True,  "凍 4 → stall");
        }

        // ★ 低線掃（合法慢速）→ 偵測窗自動拉長，不誤判：1fps（幀週期 1000ms）窗 = 4 + ceil(1500/500)=7。
        [Test]
        public void LowFps_WidensWindow_NoFalseStall()
        {
            var d = new CameraStallDetector();
            d.Update(1, 100, 1.0);                                 // priming（1fps）
            for (int i = 1; i <= 6; i++)
                Assert.That(d.Update(1, 100, 1.0), Is.False, $"低FPS第{i}凍tick不該誤判（窗=7）");
            Assert.That(d.Update(1, 100, 1.0), Is.True, "低FPS第7凍tick滿窗 → stall");
        }

        // 不同相機各自獨立累計，互不影響。
        [Test]
        public void PerCamera_IndependentAccumulation()
        {
            var d = new CameraStallDetector();
            d.Update(1, 100, 0); d.Update(2, 200, 0);              // 兩台 priming
            for (int i = 0; i < 3; i++) { d.Update(1, 100, 0); d.Update(2, 200 + i + 1, 0); } // cam1 凍、cam2 動
            Assert.That(d.Update(1, 100, 0), Is.True,  "cam1 凍住滿窗 → stall");
            Assert.That(d.Update(2, 210, 0), Is.False, "cam2 一直在動 → 不 stall");
        }
    }
}
