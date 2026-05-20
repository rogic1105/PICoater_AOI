#pragma once

#include <string>
#include <unordered_map>

#include "i_plc_adapter.hpp"

namespace picoater::plc {

class MockPlcAdapter : public IPlcAdapter {
 public:
  bool Connect() override;
  void Disconnect() override;
  bool ReadBit(int address, bool* value) override;
  bool WriteBit(int address, bool value) override;
  const std::string& GetLastError() const override;

 private:
  bool connected_ = false;
  std::unordered_map<int, bool> bit_points_;
  std::string last_error_;
};

}  // namespace picoater::plc
