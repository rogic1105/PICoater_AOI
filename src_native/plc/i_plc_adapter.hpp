#pragma once

#include <string>

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

}  // namespace picoater::plc
