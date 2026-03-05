#pragma once

#include <string>
#include <unordered_map>

namespace picoater::plc {

class IPlcAdapter {
 public:
  virtual ~IPlcAdapter() = default;

  virtual bool Connect() = 0;
  virtual void Disconnect() = 0;
  virtual bool ReadBit(int address, bool* value) = 0;
  virtual bool WriteBit(int address, bool value) = 0;
  virtual const std::string& GetLastError() const = 0;
};

class MockPlcAdapter : public IPlcAdapter {
 public:
  bool Connect() override {
    connected_ = true;
    last_error_.clear();
    return true;
  }

  void Disconnect() override {
    connected_ = false;
    last_error_.clear();
  }

  bool ReadBit(int address, bool* value) override {
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

  bool WriteBit(int address, bool value) override {
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

  const std::string& GetLastError() const override { return last_error_; }

 private:
  bool connected_ = false;
  std::unordered_map<int, bool> bit_points_;
  std::string last_error_;
};

}  // namespace picoater::plc
