#include "fa_out/backends/factory.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace
{

class FakeSinkBackend final : public fa_out::backends::SinkBackend
{
public:
  fa_out::backends::SinkOpenInfo open() override
  {
    return {};
  }

  void close() override {}

  bool isOpen() const override
  {
    return true;
  }

  bool isRunning() const override
  {
    return true;
  }

  size_t writeFrames(const uint8_t * /*data*/, size_t frame_count) override
  {
    return frame_count;
  }
};

fa_out::backends::SinkBackendSettings settingsWithName(const std::string & name)
{
  fa_out::backends::SinkBackendSettings settings;
  settings.name = name;
  return settings;
}

}  // namespace

TEST(SinkBackendFactoryTest, BuildsInjectedAlsaBackend)
{
  auto settings = settingsWithName("alsa_playback");
  settings.alsa_playback.device_id = "hw:0,0";

  bool factory_called = false;
  auto backend = fa_out::backends::buildSinkBackend(
    settings,
    [&](const fa_out::backends::AlsaPlaybackConfig & config) {
      factory_called = true;
      EXPECT_EQ(config.device_id, "hw:0,0");
      return std::make_unique<FakeSinkBackend>();
    });

  EXPECT_TRUE(factory_called);
  EXPECT_NE(backend, nullptr);
}

TEST(SinkBackendFactoryTest, BuildsPcmFileWriterBackend)
{
  auto settings = settingsWithName("pcm_file_writer");
  settings.pcm_file_writer.file_path = "/tmp/fa_out.raw";

  auto backend = fa_out::backends::buildSinkBackend(settings);

  EXPECT_NE(dynamic_cast<fa_out::backends::PcmFileWriterBackend *>(backend.get()), nullptr);
}

TEST(SinkBackendFactoryTest, BuildsNetworkPcmSenderBackend)
{
  auto settings = settingsWithName("network_pcm_sender");
  settings.network_pcm_sender.endpoint_uri = "udp://127.0.0.1:49000";
  settings.network_pcm_sender.encoding = "PCM16LE";
  settings.network_pcm_sender.channels = 1;
  settings.network_pcm_sender.bit_depth = 16;
  settings.network_pcm_sender.max_packet_bytes = 320;

  auto backend = fa_out::backends::buildSinkBackend(settings);

  EXPECT_NE(dynamic_cast<fa_out::backends::NetworkPcmSenderBackend *>(backend.get()), nullptr);
}

TEST(SinkBackendFactoryTest, RejectsMissingAlsaFactory)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_out::backends::buildSinkBackend(
        settingsWithName("alsa_playback"),
        fa_out::backends::AlsaPlaybackBackendFactory{}));
    },
    std::runtime_error);
}

TEST(SinkBackendFactoryTest, RejectsNullAlsaFactoryResult)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_out::backends::buildSinkBackend(
        settingsWithName("alsa_playback"),
        [](const fa_out::backends::AlsaPlaybackConfig &)
          -> std::unique_ptr<fa_out::backends::SinkBackend> {
          return nullptr;
        }));
    },
    std::runtime_error);
}

TEST(SinkBackendFactoryTest, RejectsMissingBackendName)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_out::backends::buildSinkBackend(
        settingsWithName(""),
        [](const fa_out::backends::AlsaPlaybackConfig &) {
          return std::make_unique<FakeSinkBackend>();
        }));
    },
    std::runtime_error);
}

TEST(SinkBackendFactoryTest, RejectsUnknownBackendName)
{
  EXPECT_THROW(
    {
      static_cast<void>(fa_out::backends::buildSinkBackend(
        settingsWithName("bogus"),
        [](const fa_out::backends::AlsaPlaybackConfig &) {
          return std::make_unique<FakeSinkBackend>();
        }));
    },
    std::runtime_error);
}
