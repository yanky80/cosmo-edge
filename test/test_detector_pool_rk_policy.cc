#include <memory>
#include <string>

#include "catch_amalgamated.hpp"
#include "util/ErrorCode.h"
#include "util/InstancePool.h"

namespace {

// Tracks live instances so the pool policy (one instance per task, at most
// three) can be verified without NPU hardware.
class CountingTask {
public:
    static int live;
    static int created;

    CountingTask(const std::string&, const std::string&, const std::string&) {
        ++created;
    }

    ~CountingTask() {
        --live;
    }

    CountingTask(const CountingTask&)            = delete;
    CountingTask& operator=(const CountingTask&) = delete;

    cosmo::util::ErrorEnum Init() {
        ++live;
        return cosmo::util::ErrorEnum::Success;
    }
};

int CountingTask::live    = 0;
int CountingTask::created = 0;

}  // namespace

TEST_CASE("RK3588 DetectorPool policy: one instance per task, at most three instances",
          "[infer][pool][rk-policy]") {
    CountingTask::live    = 0;
    CountingTask::created = 0;

    cosmo::InstancePool<CountingTask, std::shared_ptr<CountingTask>> pool("rk_detector", 1, 3);

    // Four tasks but only three instances may exist.
    pool.CreateTask("alg", "cfg", "model");
    pool.CreateTask("alg", "cfg", "model");
    pool.CreateTask("alg", "cfg", "model");
    pool.CreateTask("alg", "cfg", "model");
    CHECK(CountingTask::live == 3);
    CHECK(CountingTask::created == 3);

    // Three tasks can hold an instance simultaneously; the fourth times out.
    auto first  = pool.GetInst("alg", "cfg", "model", 100);
    auto second = pool.GetInst("alg", "cfg", "model", 100);
    auto third  = pool.GetInst("alg", "cfg", "model", 100);
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    REQUIRE(third != nullptr);
    auto fourth = pool.GetInst("alg", "cfg", "model", 100);
    CHECK(fourth == nullptr);

    pool.ReturnInst(first);
    pool.ReturnInst(second);
    pool.ReturnInst(third);
    CHECK(CountingTask::live == 3);
}

TEST_CASE("RK3588 DetectorPool policy: deleting a task shrinks only when spare capacity exists",
          "[infer][pool][rk-policy]") {
    CountingTask::live    = 0;
    CountingTask::created = 0;

    cosmo::InstancePool<CountingTask, std::shared_ptr<CountingTask>> pool("rk_detector", 1, 3);
    pool.CreateTask("alg", "cfg", "model");
    pool.CreateTask("alg", "cfg", "model");
    pool.CreateTask("alg", "cfg", "model");
    pool.CreateTask("alg", "cfg", "model");
    CHECK(CountingTask::live == 3);

    pool.DeleteTask();  // Still three tasks of work: no instance is freed.
    CHECK(CountingTask::live == 3);

    pool.DeleteTask();
    pool.DeleteTask();
    pool.DeleteTask();  // No tasks left: instances may be released.
    CHECK(CountingTask::live == 0);
}
