#include "fa_out/backends/sink_backend.hpp"

namespace fa_out::backends
{

SinkBackendError::SinkBackendError(const std::string & message)
: std::runtime_error(message)
{
}

}  // namespace fa_out::backends
