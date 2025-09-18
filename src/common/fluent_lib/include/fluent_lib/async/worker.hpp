#pragma once

#include <thread>
#include <atomic>
#include <functional>

namespace fluent_lib::async {

class Worker {
public:
    template <class Fn>
    explicit Worker(Fn &&fn) { th_ = std::thread(std::forward<Fn>(fn)); }
    ~Worker() { if (th_.joinable()) th_.join(); }
private:
    std::thread th_;
};

} // namespace fluent_lib::async

