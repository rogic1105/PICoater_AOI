#pragma once
// 極簡 flat-json 取值（API 參數專用，零外部依賴）。
// 只支援單層物件 {"key": 1.5, "key2": "str"}；key 唯一、無跳脫、無巢狀 —— C# 端組的參數 json 固定此格式。
// 多 pipeline 各自 parse 自己要的 key，加參數不破 C ABI（= 4b 設計：run(name, json) 單一簽名）。
#include <cstdlib>
#include <cstring>
#include <string>

namespace tanuki { namespace pipeline { namespace jsonlite {

// 找 "key" 的值起點（冒號後第一個非空白字元）；找不到回 nullptr。
inline const char* find_value(const char* json, const char* key) {
    if (!json || !key) return nullptr;
    std::string quoted = std::string("\"") + key + "\"";
    const char* p = std::strstr(json, quoted.c_str());
    if (!p) return nullptr;
    p += quoted.size();
    while (*p == ' ' || *p == '\t') ++p;
    if (*p != ':') return nullptr;
    ++p;
    while (*p == ' ' || *p == '\t') ++p;
    return p;
}

// 取數值；缺 key 用 fallback。
inline float get_number(const char* json, const char* key, float fallback) {
    const char* v = find_value(json, key);
    return v ? (float)std::atof(v) : fallback;
}

// 取字串（不含引號）；缺 key 用 fallback。
inline std::string get_string(const char* json, const char* key, const char* fallback) {
    const char* v = find_value(json, key);
    if (!v || *v != '"') return fallback ? fallback : "";
    ++v;
    const char* e = std::strchr(v, '"');
    return e ? std::string(v, e - v) : (fallback ? fallback : "");
}

}}} // namespace tanuki::pipeline::jsonlite
