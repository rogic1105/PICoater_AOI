//AOI_SDK\framework\include\framework\test_utils.hpp

#pragma once

#include <functional>
#include <string>

namespace framework {

    // 定義測試函式的標準簽名
    using TestEntryFunc = std::function<void(const std::string&)>;

    // 取得統一的測試輸出路徑
    // 會自動建立資料夾: artifacts/<suiteName>/
    // 回傳: 完整的檔案絕對路徑
    std::string GetOutputPath(const std::string& suiteName, const std::string& fileName);

    // 通用的測試啟動器
    int RunAOITestBootstrap(const std::string& suiteName, TestEntryFunc testFunc);

}  // namespace framework