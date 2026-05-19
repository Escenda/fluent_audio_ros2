#include "fa_kws/backend_config_validation.hpp"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace
{

void expectRuntimeError(const std::string & backend_name, const std::string & expected_message)
{
  try {
    fa_kws::validation::requireSupportedBackendName(backend_name);
    FAIL() << "expected runtime_error: " << expected_message;
  } catch (const std::runtime_error & error) {
    EXPECT_STREQ(expected_message.c_str(), error.what());
  }
}

}  // namespace

TEST(KwsBackendConfigValidationTest, AcceptsSherpaOnnxKwsBackendName)
{
  EXPECT_NO_THROW(
    fa_kws::validation::requireSupportedBackendName(fa_kws::validation::kBackendSherpaOnnxKws));
}

TEST(KwsBackendConfigValidationTest, RejectsMissingBackendName)
{
  expectRuntimeError("", "backend.name is required");
}

TEST(KwsBackendConfigValidationTest, RejectsUnknownBackendName)
{
  expectRuntimeError("bogus", "unsupported fa_kws backend.name: bogus");
}
