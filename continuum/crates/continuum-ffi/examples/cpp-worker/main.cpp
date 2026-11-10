// An example C++ worker that serves a job the broker dispatched, end to end. Usage: cpp_worker <broker socket>
//
// One process plays three parts, on two connections:
// - A producer writes three camera frames into Tier 0 slots (claim, fill, publish, ingest) and confirms each
//   commit through a live subscription.
// - A consumer asks for poses over those frames. Nothing has made them yet, so the broker quotes the missing part
//   and the consumer commits the promise.
// - The worker, which registered that it computes poses from the camera, gets the dispatched job. It reads each
//   input frame in place under a pin (released when the pin goes out of scope) and commits one pose on the job's
//   output lease. The consumer's promise then delivers it.
// Exits 0 if all of that worked.
//
// It shows both error styles: exceptions (Client::connect and friends throw continuum::Error) and the noexcept
// forms that return a continuum::Status.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "continuum.hpp"

namespace {

constexpr const char* kCam = "example.cpp.cam";
constexpr const char* kPose = "example.cpp.pose";

[[noreturn]] void fail(const std::string& what) {
    std::fprintf(stderr, "%s\n", what.c_str());
    std::exit(1);
}

void check(const continuum::Status& s, const char* what) {
    if (!s) fail(std::string(what) + ": " + s.message);
}

/// The index (in Welcome order) of the arena a location names.
std::size_t arena_index(const continuum::Client& client, const cnt_location& at) {
    std::vector<continuum::ArenaInfo> arenas = client.arenas();
    for (std::size_t i = 0; i < arenas.size(); ++i) {
        if (arenas[i].name == at.arena) return i;
    }
    fail(std::string("no arena named ") + at.arena);
}

/// Writes three frames over [100, 400) into Tier 0 slots and ingests them, waiting for each commit.
void produce(continuum::Client& client, std::uint32_t cam) {
    continuum::Subscription live;
    check(client.subscribe({kCam}, CNT_START_LIVE, live), "subscribe");
    for (std::uint64_t n = 0; n < 3; ++n) {
        continuum::WriteSlot slot = client.claim_slot(0, 0, cam);
        std::string frame = "frame " + std::to_string(n);
        std::memcpy(slot.bytes().data(), frame.data(), frame.size());
        cnt_location at{};
        check(slot.publish(frame.size(), at), "publish");

        cnt_header_spec h{};
        continuum::Id id = continuum::new_id();
        std::memcpy(h.id, id.data(), id.size());
        h.stream_id = cam;
        h.start_ns = 100 + 100 * n;
        h.end_ns = h.start_ns + 100;
        h.layer = 0;   // Layer1Raw
        h.format = 0;  // RawBytes
        h.shape_name = "frame";
        h.shape_version = 1;
        std::vector<std::uint8_t> header = continuum::build_header(h);
        check(client.ingest(header, &at), "ingest");
    }
    for (int committed = 0; committed < 3;) {
        continuum::SubEvent e;
        check(live.next(5000, e), "waiting for the frames' commits");
        if (e.type == CNT_SUB_EVENT) ++committed;
    }
    std::printf("produced 3 frames into Tier 0 slots\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cpp_worker <broker socket>\n");
        return 2;
    }
    try {
        continuum::Client worker = continuum::Client::connect(argv[1], {.client_name = "cpp-example"});
        std::printf("connected; broker clock %llu ns\n", static_cast<unsigned long long>(worker.heartbeat()));

        continuum::StreamSpec cam_spec;
        cam_spec.name = kCam;
        cam_spec.kind = CNT_SENSOR;
        cam_spec.shape_name = "frame";
        std::uint32_t cam = worker.register_stream(cam_spec).stream_id;

        continuum::StreamSpec pose_spec;
        pose_spec.name = kPose;
        pose_spec.kind = CNT_DERIVED;
        pose_spec.format = 4;  // FlatBuffer
        pose_spec.layer = 1;   // Layer2Decision
        pose_spec.shape_name = "pose";
        std::uint32_t pose = worker.register_stream(pose_spec).stream_id;

        // The noexcept form: a conflicting registration comes back as a Status, not an exception.
        continuum::StreamSpec other = pose_spec;
        other.shape_name = "not-pose";
        cnt_stream_registered ignored{};
        continuum::Status conflict = worker.register_stream(other, ignored);
        if (conflict || conflict.code != CNT_BROKER) fail("expected a broker refusal for the conflicting shape");
        std::printf("conflicting shape refused (broker code %u)\n", conflict.broker_code);

        cnt_requirement needs{};
        needs.stream = kCam;
        cnt_capability cap{};
        cap.produces = kPose;
        cap.inputs = &needs;
        cap.n_inputs = 1;
        cap.compute_domain = 0;  // CPU
        cap.max_batch = 1;
        cap.preemptible = 1;
        cap.outputs_per_job = 1;
        std::vector<cnt_capability_result> results;
        std::string reasons;
        continuum::Status reg = worker.register_worker("worker:cpp-example", {&cap, 1}, results, &reasons);
        if (!reg || !results[0].accepted || !results[0].runnable) fail("worker registration: " + reg.message + reasons);
        std::printf("registered as a worker: %s from %s\n", kPose, kCam);

        produce(worker, cam);

        // The consumer: what's there for poses over the frames? Nothing yet, so the broker quotes it.
        continuum::Client consumer = continuum::Client::connect(argv[1], {.client_name = "cpp-consumer"});
        cnt_query_spec q{};
        q.stream = kPose;
        q.range_start_ns = 100;
        q.range_end_ns = 400;
        q.mode = 1;  // QuoteIfMissing
        q.band = 2;  // P2Nearline
        continuum::QueryResult found = consumer.query(q);
        cnt_promise_quote quote{};
        if (!found.items().empty() || !found.quote(quote)) fail("expected no poses and a quote for them");
        continuum::Id promise_id{};
        std::memcpy(promise_id.data(), quote.promise_id, promise_id.size());
        continuum::Promise promise = consumer.commit_promise(promise_id);
        std::printf("quoted about %.0f ms for the missing poses; committed the promise\n",
                    static_cast<double>(quote.est_latency_ms));

        // The worker: the dispatched job.
        continuum::Event event;
        do {
            check(worker.next_event(5000, event), "waiting for the job");
        } while (event.raw.type != CNT_EVENT_DISPATCH_JOB);
        continuum::Job job = std::move(event.job);
        cnt_job_info info = job.info();
        continuum::WorkLease lease = worker.take_lease(job);

        std::vector<std::uint8_t> parents;
        continuum::ItemsView inputs = job.inputs();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            cnt_item item = inputs[i];
            cnt_header_info in = continuum::read_header(inputs.header(i));
            parents.insert(parents.end(), in.id, in.id + 16);
            if (item.at.kind != CNT_LOCATION_SHM) fail("an input isn't in a Tier 0 slot");
            // Read the frame in place under a pin; the pin is released when `pin` goes out of scope.
            cnt_slot_ref ref{item.at.slab_class, item.at.slot, item.at.slot_generation};
            continuum::Lease pin = worker.lease(arena_index(worker, item.at), ref);
            std::string_view frame(reinterpret_cast<const char*>(pin.bytes().data()), pin.bytes().size());
            std::printf("input %zu at %llu ns: \"%.*s\"\n", i, static_cast<unsigned long long>(in.start_ns),
                        static_cast<int>(frame.size()), frame.data());
        }

        cnt_header_spec out{};
        continuum::Id out_id = continuum::new_id();
        std::memcpy(out.id, out_id.data(), out_id.size());
        out.stream_id = pose;
        out.start_ns = info.output_start_ns;
        out.end_ns = info.output_end_ns;
        out.layer = 1;   // Layer2Decision
        out.format = 4;  // FlatBuffer
        out.shape_name = "pose";
        out.shape_version = 1;
        out.parents = parents.data();
        out.n_parents = inputs.size();
        std::vector<std::uint8_t> header = continuum::build_header(out);
        const char body[] = "pose";
        cnt_location at{};
        at.kind = CNT_LOCATION_INLINE;
        at.inline_bytes = reinterpret_cast<const std::uint8_t*>(body);
        at.inline_len = sizeof body - 1;
        cnt_output output{header.data(), header.size(), &at};
        check(lease.commit({&output, 1}), "commit");
        std::printf("served the job: one pose over [%llu, %llu) from %zu frames\n",
                    static_cast<unsigned long long>(info.output_start_ns),
                    static_cast<unsigned long long>(info.output_end_ns), inputs.size());

        // The consumer's promise delivers it.
        std::vector<continuum::Items> parts;
        check(promise.fulfilled(5000, parts), "waiting for the promise");
        std::size_t delivered = 0;
        bool ours = false;
        for (const continuum::Items& part : parts) {
            for (std::size_t i = 0; i < part.size(); ++i, ++delivered) {
                cnt_header_info h = continuum::read_header(part.view().header(i));
                ours = ours || std::memcmp(h.id, out_id.data(), 16) == 0;
            }
        }
        if (delivered != 1 || !ours) fail("the promise didn't deliver the pose");
        std::printf("the promise delivered the pose\n");
        return 0;  // each Client's destructor says Goodbye
    } catch (const continuum::Error& e) {
        std::fprintf(stderr, "continuum error %d: %s\n", static_cast<int>(e.code()), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
