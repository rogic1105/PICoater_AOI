#include "mock_plc_adapter.hpp"

namespace picoater::plc {

bool MockPlcAdapter::Connect() {
  connected_ = true;
  last_error_.clear();
  return true;
}

void MockPlcAdapter::Disconnect() {
  connected_ = false;
  last_error_.clear();
}

bool MockPlcAdapter::ReadBit(int address, bool* value) {
  if (value == nullptr) {
    last_error_ = "ReadBit value output pointer must not be null.";
    return false;
  }

  if (!connected_) {
    last_error_ = "PLC adapter is not connected.";
    return false;
  }

  if (address < 0) {
    last_error_ = "PLC bit address must be non-negative.";
    return false;
  }

  const auto it = bit_points_.find(address);
  *value = it != bit_points_.end() ? it->second : false;
  last_error_.clear();
  return true;
}

bool MockPlcAdapter::WriteBit(int address, bool value) {
  if (!connected_) {
    last_error_ = "PLC adapter is not connected.";
    return false;
  }

  if (address < 0) {
    last_error_ = "PLC bit address must be non-negative.";
    return false;
  }

  bit_points_[address] = value;
  last_error_.clear();
  return true;
}

const std::string& MockPlcAdapter::GetLastError() const { return last_error_; }

}  // namespace picoater::plc
