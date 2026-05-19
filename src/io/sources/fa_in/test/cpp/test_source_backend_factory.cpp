#include "fa_in/backends/factory.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fa_in/backends/network_pcm_receiver_backend.hpp"
#include "fa_in/backends/pcm_file_reader_backend.hpp"

namespace
{

class FakeSourceBackend final : public fa_in::backends::SourceBackend
{
public:
  std::vector<fa_in::backends::DeviceInfo> listDevices() const override
  {
    return {};
  }

  fa_in::backends::DeviceInfo selectDevice(
    const fa_in::backends::DeviceSelector & /*selector*/) const override
  {
    return {};
  }

  size_t open(
    const std::string & /*device_id*/,
    const fa_in::backends::AudioFormat & /*format*/,
    size_t requested_frames) override
  {
    return requested_frames;
  }

  void close() override {}

  void drop() override {}

  fa_in::backends::ReadResult read(uint8_t * /*data*/, size_t /*frames*/) override
  {
    return {};
  }
};

fa_in::backends::SourceBackendSettings settingsWithName(const std::string & name)
{
  fa_in::backends::SourceBackendSettings settings;
  settings.name = name;
  return settings;
}

}  // namespace

TEST(SourceBackendFactoryTest, BuildsInjectedAlsaBackend)
{
  auto backend = fa_in::backends::buildSourceBackend(
    settingsWithName("alsa_capture"),
    []() {
      return std::make_unique<FakeSourceBackend>();
    });

  EXPECT_NE(backend, nullptr);
}

TEST(SourceBackendFactoryTest, BuildsPcmFileReaderBackend)
{
  auto backend = fa_in::backends::buildSourceBackend(settingsWithName("pcm_file_reader"));

  EXPECT_NE(dynamic_cast<fa_in::backends::PcmFileReaderBackend *>(backend.get()), nullptr);
}

TEST(SourceBackendFactoryTest, BuildsNetworkPcmReceiverBackend)
{
  auto backend = fa_in::backends::buildSourceBackend(settingsWithName("network_pcm_receiver"));

  EXPECT_NE(dynamic_cast<fa_in::backends::NetworkPcmReceiverBackend *>(backend.get()), nullptr);
}

TEST(SourceBackendFactoryTest, RejectsMissingAlsaFactory)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_in::backends::buildSourceBackend(
        settingsWithName("alsa_capture"),
        fa_in::backends::SourceBackendFactory{}));
    },
    std::runtime_error);
}

TEST(SourceBackendFactoryTest, RejectsNullAlsaFactoryResult)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_in::backends::buildSourceBackend(
        settingsWithName("alsa_capture"),
        []() -> std::unique_ptr<fa_in::backends::SourceBackend> {
          return nullptr;
        }));
    },
    std::runtime_error);
}

TEST(SourceBackendFactoryTest, RejectsMissingBackendName)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_in::backends::buildSourceBackend(
        settingsWithName(""),
        []() {
          return std::make_unique<FakeSourceBackend>();
        }));
    },
    std::runtime_error);
}

TEST(SourceBackendFactoryTest, RejectsUnknownBackendName)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_in::backends::buildSourceBackend(
        settingsWithName("bogus"),
        []() {
          return std::make_unique<FakeSourceBackend>();
        }));
    },
    std::runtime_error);
}
