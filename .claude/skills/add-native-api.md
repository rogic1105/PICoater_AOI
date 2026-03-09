# add-native-api

新增或修改 Native C API 函式（C++ 實作 + C# P/Invoke）

## 使用時機

當需要：
- 在 `core_cv_api.dll` 新增 GPU 函式
- 在 `picoater_api.dll` 新增 pipeline 函式
- 更新 C# 端的 P/Invoke 宣告

## 執行步驟

1. **C 標頭宣告** — `sdk/AOI_SDK/core_cv_api/include/export_c/export_api.h`
   - 加入 `CORE_CV_API` 修飾的函式簽章

2. **C++ 實作** — `sdk/AOI_SDK/core_cv_api/src/export_api.cpp`
   - 實作函式，使用 CUDA kernel 或 core_cv 內部 API
   - 若有 GPU kernel：allocate d_buf → H2D → kernel → sync → D2H → free

3. **C# P/Invoke** — `AniloxRoll.Monitor/Interop/NativeMethods.cs`
   - 加在對應 DLL 區塊（`CoreCVDllName` 或 `DllName`）
   - **不得** 修改 `sdk/AOI_SDK/src_dotnet/AOI.SDK/Core/CoreCVWrapper.cs` 來供 Monitor 使用

4. **驗證**：確認 `CallingConvention.Cdecl`、`MarshalAs(UnmanagedType.I1)` (bool)、`[MarshalAs(UnmanagedType.LPStr)]` (string) 正確

## 範本

```csharp
// NativeMethods.cs — core_cv_api.dll 區塊
[DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
public static extern int CoreCV_NewFunction(IntPtr src, int w, int h, IntPtr dst);
```

```cpp
// export_api.cpp
CORE_CV_API int CoreCV_NewFunction(const uint8_t* h_src, int w, int h, uint8_t* h_dst)
{
    uint8_t* d_buf = nullptr;
    cudaMalloc(&d_buf, (size_t)w * h);
    cudaMemcpy(d_buf, h_src, (size_t)w * h, cudaMemcpyHostToDevice);
    // ... kernel call ...
    cudaDeviceSynchronize();
    cudaMemcpy(h_dst, d_buf, (size_t)w * h, cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    return 0;
}
```
