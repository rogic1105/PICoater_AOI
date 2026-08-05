---
name: add-test
description: Add or reorganize automated tests in the PICoater AOI repository. Use when deciding between unit, integration, stress, UI, or DVT log checks and when wiring new test projects.
---

# add-test

## Verification platform command vocabulary

Use [`references/verification-platform.md`](references/verification-platform.md) as the SSoT for
the short commands `功能測試`, `完整驗證`, `壓力測試`, and `耐久測試`. Do not use the ambiguous
terms `外掛` or `內掛` for test infrastructure.

新增測試時遵循的分類 + 模板。

## 使用時機

寫任何新測試（單元 / 整合 / 壓力）前，先決定該歸哪一層。

長時間 load、soak 或失效注入另讀 [`references/stress-and-soak.md`](references/stress-and-soak.md)。

## 三層測試 csproj

| 類型 | 位置 | 何時寫 |
|---|---|---|
| **Unit** | `tests/AniloxRoll.Monitor.Tests/` | 純邏輯、無 IO、Mock 對外（< 5ms / case，每次 commit 都該過） |
| **Integration** | `tests/AniloxRoll.Monitor.Integration.Tests/` | 檔案 IO、JSON 讀寫、Mock 硬體（< 1s / case，PR / nightly 跑） |
| **Stress** | `tests/AniloxRoll.Monitor.Stress.Tests/` | 長迴圈、Soak、Load（數十秒到小時，週期跑） |

## 決定該寫哪一層

```
新測試 → 需要 Path.GetTempFileName / File.* / JSON IO？
        → 是 → Integration
        → 否 → 跑很久（> 1s）或長迴圈？
              → 是 → Stress
              → 否 → Unit
```

## 範例對照

### Unit（純 mock）
```csharp
// tests/AniloxRoll.Monitor.Tests/IoGrabControllerTests.cs
[Test]
public void NotifyGrabStarted_SetsBusyDoHigh()
{
    var mock = new Mock<IModbusTcpClient>();
    var ctrl = new IoGrabController(mock.Object, ...);
    ctrl.NotifyGrabStarted();
    mock.Verify(c => c.WriteSingleCoil(DO_PC_BUSY, true), Times.Once);
}
```

### Integration（檔案 IO）
```csharp
// tests/AniloxRoll.Monitor.Integration.Tests/InspectionLogServiceTests.cs
[Test]
public void AppendRecord_CreatesCsvWithCfgHeader()
{
    string tmpDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    Directory.CreateDirectory(tmpDir);
    try
    {
        var svc = new InspectionLogService(() => tmpDir);
        svc.AppendRecord(...);
        Assert.That(File.Exists(...), Is.True);
    }
    finally { Directory.Delete(tmpDir, true); }
}
```

### Stress（長迴圈）
```csharp
// tests/AniloxRoll.Monitor.Stress.Tests/StressTests.cs
[Test]
[Category("Stress")]
public void Settings_RoundTrip_StressTest()
{
    int minutes = int.TryParse(Environment.GetEnvironmentVariable("STRESS_MINUTES"), out var m) ? m : 1;
    var sw = Stopwatch.StartNew();
    int count = 0;
    while (sw.Elapsed.TotalMinutes < minutes)
    {
        // ... 大量 round-trip
        count++;
    }
    TestContext.WriteLine($"completed {count} iterations in {sw.Elapsed.TotalSeconds:F1}s");
}
```

## 新增第四個測試 csproj 時

1. 創 `tests/AniloxRoll.Monitor.{X}.Tests/{X}.Tests.csproj` — 複製現有 csproj 為模板
2. 設定 `<RootNamespace>` / `<AssemblyName>`
3. `Properties/AssemblyInfo.cs` 加 `[assembly: AssemblyTitle("...")]` + 新 GUID
4. **`src/dotnet/AniloxRoll.Monitor/Properties/AssemblyInfo.cs` 加 `[assembly: InternalsVisibleTo("AniloxRoll.Monitor.{X}.Tests")]`** ← **必要**
5. `PICoater_AOI.sln` 加 Project entry + Release|x64 config
6. ProjectReference 引用 `src/dotnet/AniloxRoll.Monitor/AniloxRoll.Monitor.csproj`（其他 sdk 視需要）

## 反模式

- ❌ Unit 測試內 `File.WriteAllText` — 應該 Integration
- ❌ Integration 內 `Thread.Sleep(60_000)` 或長迴圈 — 應該 Stress
- ❌ 一個 csproj 內混三類 — CI 無法分別跑（之前的設計，已拆）
- ❌ 用 internal 但沒加 `InternalsVisibleTo` — CS0117 編譯錯
- ❌ 測試名稱沒寫意圖（`Test1()`、`MyTest()`）— 用 `Method_Scenario_Expected` 慣例
