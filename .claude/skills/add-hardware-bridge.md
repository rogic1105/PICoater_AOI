# add-hardware-bridge

新增「硬體 bridge」（PLC、光源、儲存、相機 SDK、CLProtocol grabber 等任何外部硬體/服務的通訊封裝）時遵循的結構模板。

## 使用時機

需要為新的硬體 / 通訊協定 / 外部服務寫適配層（adapter）時。例如：
- 新光源型號（不同 RS-232 protocol）
- 新 PLC 廠牌（非 ICP DAS / Modbus TCP）
- 新儲存方案（不是 SMB，改 S3 / FTP）
- 相機 SDK（如 Basler、FLIR）

## 黃金規則（業界 monorepo 共識）

**library 跟 executable 必須實體分離**：
- `sdk/{XBridge}/{XBridge.Core}/` — **純 library**，無 GUI 無 exe
- `tools/x-manual-control/`、`tools/x-automation/` — **GUI 工具**（WinForms / WPF / CLI exe）

**依賴方向**：`src/AniloxRoll.Monitor → sdk/{XBridge}/Core`；**Bridge 絕對不能反向依賴 src/Monitor 內任何東西**（包括 InspectionSettings）。

## 模板（以新增 ExampleBridge 為例）

### 1. 建 sdk/ 目錄結構

```
sdk/ExampleBridge/
└── ExampleBridge.Core/
    ├── ExampleBridge.Core.csproj
    ├── IExampleClient.cs           ← 介面（給 Monitor mock 注入測試）
    ├── ExampleClient.cs            ← 實作
    ├── ExampleLogger.cs            ← 內部 logger（可選）
    └── Properties/AssemblyInfo.cs
```

### 2. csproj 模板（複製 sdk/Bridges/IoBridge/IoBridge.Core/IoBridge.Core.csproj）

關鍵欄位：
```xml
<ProjectGuid>{ 新 GUID }</ProjectGuid>
<OutputType>Library</OutputType>
<RootNamespace>ExampleBridge.Core</RootNamespace>
<AssemblyName>ExampleBridge.Core</AssemblyName>
<TargetFrameworkVersion>v4.8</TargetFrameworkVersion>
```

只 reference `System` / `System.Core` + 真正需要的（如 `System.IO.Ports` for RS-232）。**不 reference** Monitor 內任何 namespace。

### 3. namespace 規範

```csharp
namespace ExampleBridge.Core
{
    public interface IExampleClient { ... }
    public class ExampleClient : IExampleClient { ... }
}
```

caller (`Monitor`) 加 `using ExampleBridge.Core;`。

### 4. API 設計 — 解耦 InspectionSettings

**禁止**：
```csharp
// ❌ Bridge 內知道 InspectionSettings 存在 → 反向依賴 src/
public ExampleClient(InspectionSettings settings) { ... }
```

**正確**：
```csharp
// ✓ 接基本參數，caller (Monitor) 自己從 InspectionSettings 拿值傳進來
public ExampleClient(string comPort, int baudRate, int timeoutMs) { ... }
```

### 5. AniloxRoll.Monitor.csproj 加 ProjectReference

```xml
<ProjectReference Include="..\..\..\sdk\ExampleBridge\ExampleBridge.Core\ExampleBridge.Core.csproj">
  <Project>{ 新 GUID }</Project>
  <Name>ExampleBridge.Core</Name>
</ProjectReference>
```

### 6. PICoater_AOI.sln 加 entries（兩處）

a) Project 區段：
```
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "ExampleBridge.Core", "sdk\ExampleBridge\ExampleBridge.Core\ExampleBridge.Core.csproj", "{ 新 GUID }"
EndProject
```

b) GlobalSection(ProjectConfigurationPlatforms) 區段：
```
{ 新 GUID }.Release|x64.ActiveCfg = Release|x64
{ 新 GUID }.Release|x64.Build.0 = Release|x64
```

### 7. tools/ 工具（如果有）

GUI / exe / 腳本放 `tools/x-manual-control/`、`tools/x-automation/`。命名規範：小寫連字號（與 sdk/ 大駝峰區分）。

tools 內 csproj 透過 ProjectReference 引用 sdk：
```xml
<ProjectReference Include="..\..\..\sdk\ExampleBridge\ExampleBridge.Core\ExampleBridge.Core.csproj" />
```

### 8. Monitor 整合

在 `src/dotnet/AniloxRoll.Monitor/Services/` 內建 `XGrabController` 或類似業務邏輯類，**組合** Bridge（注入 `IExampleClient`）+ FSM / 業務規則。Bridge 只負責通訊，業務邏輯不進 sdk。

```csharp
// src/dotnet/AniloxRoll.Monitor/Services/ExampleGrabController.cs
public class ExampleGrabController
{
    private readonly IExampleClient _client;
    public ExampleGrabController(IExampleClient client) { _client = client; }
    // ...FSM、業務邏輯
}
```

## 既有實例（複製模板用）

- `sdk/Bridges/IoBridge/IoBridge.Core/` — Modbus TCP（含 `IModbusTcpClient` 介面）
- `sdk/Bridges/LightBridge/LightBridge.Core/` — LTS-3DPA24 RS-232
- `sdk/Bridges/StorageBridge/StorageBridge.Core/` — SMB 檔案複製 + 循環儲存 + cleanup flag

## 命名規範

| 元件 | 大小寫 | 範例 |
|---|---|---|
| sdk/ 子目錄 | 大駝峰 | `IoBridge`、`LightBridge` |
| sdk/ csproj | 大駝峰 + `.Core` | `IoBridge.Core` |
| tools/ 子目錄 | 小寫連字號 | `io-manual-control`、`io-automation` |
| namespace | 跟 csproj 同 | `IoBridge.Core`、`LightBridge.Core` |

## 反模式

- ❌ Bridge 內 `using AniloxRoll.Monitor.Settings;` — 反向依賴
- ❌ GUI exe 放 sdk/ — 引用 sdk 的專案被迫拉 UI 依賴
- ❌ Bridge 內讀 `_settings.X` — 應接基本參數
- ❌ Bridge 內 `OnInspectionResult` 等業務 callback — 業務邏輯該在 Monitor
- ❌ 一個 Bridge 多個 csproj 混 library 跟 exe — 拆 sdk + tools
