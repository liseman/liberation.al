// The Continuum Broker C++20 wrapper: header-only, over the C ABI in continuum.h (crates/continuum-ffi).
//
// Errors come two ways, so code built without exceptions can use it too:
// - Every call has a form that returns a `continuum::Status` (code, broker code, message) and is noexcept.
// - When exceptions are on, the convenient forms throw `continuum::Error` (which carries the same Status).
//
// Ownership is RAII: a `Client` closes its session (Goodbye) and frees itself when destroyed, a `Lease` releases
// its slot's pin when destroyed, a `WorkLease` gives its work lease back when destroyed unless it was committed or
// aborted, and a `WriteSlot` gives an unpublished slot back. `QueryResult`, `Items`, `Promise`, `Subscription`,
// `Job`, and `Event` free what they hold. All are move-only. A Lease keeps the arena mapped, so it may outlive its
// Client; the others are used while their Client lives.
// A Client is for one thread at a time; guard it with a mutex to share it.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__cpp_exceptions)
#include <stdexcept>
#endif

#include "continuum.h"

namespace continuum {

/// The ABI this header was written against; Client::connect checks the library matches.
inline constexpr std::uint32_t abi_version = CNT_ABI_VERSION;

/// How a call went.
struct Status {
    cnt_status code = CNT_OK;
    /// The broker's ErrorCode when code is CNT_BROKER, else 0.
    std::uint16_t broker_code = 0;
    std::string message;

    /// True if the call worked.
    explicit operator bool() const noexcept { return code == CNT_OK; }

    /// The status of the last call on this thread that returned `code`.
    static Status last(cnt_status code) {
        if (code == CNT_OK) return {};
        return Status{code, cnt_last_broker_code(), cnt_last_error_message()};
    }
};

#if defined(__cpp_exceptions)
/// A failed call, when exceptions are on.
class Error : public std::runtime_error {
public:
    explicit Error(Status status) : std::runtime_error(status.message), status_(std::move(status)) {}
    const Status& status() const noexcept { return status_; }
    cnt_status code() const noexcept { return status_.code; }

private:
    Status status_;
};

namespace detail {
inline void check(const Status& s) {
    if (!s) throw Error(s);
}
}  // namespace detail
#endif

/// A payload id.
using Id = std::array<std::uint8_t, 16>;

/// A new payload id (a UUIDv7: time ordered).
inline Id new_id() noexcept {
    Id id{};
    cnt_new_id(id.data());
    return id;
}

/// Builds a PayloadHeader FlatBuffer (for Client::ingest or a commit's output) into out.
inline Status build_header(const cnt_header_spec& spec, std::vector<std::uint8_t>& out) {
    std::size_t len = 0;
    out.resize(256);
    cnt_status s = cnt_header_build(&spec, out.data(), out.size(), &len);
    if (s == CNT_INVALID_ARGUMENT && len > out.size()) {
        out.resize(len);
        s = cnt_header_build(&spec, out.data(), out.size(), &len);
    }
    out.resize(s == CNT_OK ? len : 0);
    return Status::last(s);
}

/// Reads a PayloadHeader FlatBuffer (like an item's header).
inline Status read_header(std::span<const std::uint8_t> header, cnt_header_info& out) noexcept {
    return Status::last(cnt_header_read(header.data(), header.size(), &out));
}

/// A pin on a published slot. Its bytes stay put until it's destroyed or released (or its time runs out: check
/// valid() after reading if the reader may have stalled).
class Lease {
public:
    Lease() noexcept = default;
    explicit Lease(cnt_lease* lease) noexcept : lease_(lease) {}
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept : lease_(std::exchange(other.lease_, nullptr)) {}
    Lease& operator=(Lease&& other) noexcept {
        if (this != &other) {
            reset();
            lease_ = std::exchange(other.lease_, nullptr);
        }
        return *this;
    }
    ~Lease() { reset(); }

    /// The leased bytes, read-only and 16 KB aligned. Empty for an empty Lease.
    std::span<const std::byte> bytes() const noexcept {
        if (!lease_) return {};
        return {reinterpret_cast<const std::byte*>(cnt_lease_data(lease_)), cnt_lease_len(lease_)};
    }
    /// Whether the lease still holds its slot.
    bool valid() const noexcept { return lease_ && cnt_lease_valid(lease_); }
    /// Whether this holds a lease at all.
    explicit operator bool() const noexcept { return lease_ != nullptr; }

    /// Releases the pin now. CNT_STALE if the lease had already lapsed; the Lease is empty afterward either way.
    Status release() noexcept {
        if (!lease_) return {};
        return Status::last(cnt_lease_release(std::exchange(lease_, nullptr)));
    }

private:
    void reset() noexcept {
        if (lease_) cnt_lease_release(std::exchange(lease_, nullptr));
    }
    cnt_lease* lease_ = nullptr;
};

/// A granted work lease (from Client::quote). End it with commit or abort; destroying it unended gives it back.
/// Use it while its Client lives.
class WorkLease {
public:
    WorkLease() noexcept = default;
    explicit WorkLease(cnt_work_lease* lease) noexcept : lease_(lease) {}
    WorkLease(const WorkLease&) = delete;
    WorkLease& operator=(const WorkLease&) = delete;
    WorkLease(WorkLease&& other) noexcept : lease_(std::exchange(other.lease_, nullptr)) {}
    WorkLease& operator=(WorkLease&& other) noexcept {
        if (this != &other) {
            release();
            lease_ = std::exchange(other.lease_, nullptr);
        }
        return *this;
    }
    ~WorkLease() { release(); }

    /// Whether this holds a lease.
    explicit operator bool() const noexcept { return lease_ != nullptr; }
    /// The lease's id (what a CNT_EVENT_LEASE_PREEMPTED names); 0 when empty.
    std::uint64_t id() const noexcept { return cnt_work_lease_id(lease_); }
    /// When it expires, UTC ns; 0 when empty.
    std::uint64_t expires_at_ns() const noexcept { return cnt_work_lease_expires_at_ns(lease_); }

    /// Renews by extend_ms (the broker caps it); writes the new expiry.
    Status renew(std::uint32_t extend_ms, std::uint64_t& expires_at_ns) noexcept {
        return Status::last(cnt_work_lease_renew(lease_, extend_ms, &expires_at_ns));
    }
    /// Renews in the background every every_ms (at least 10) until the lease ends.
    Status keep_alive(std::uint32_t every_ms, std::uint32_t extend_ms = 30000) noexcept {
        return Status::last(cnt_work_lease_keep_alive(lease_, every_ms, extend_ms));
    }
    /// Commits the job and returns once the outputs are durable. Afterward the WorkLease is empty, except on a bad
    /// argument (nothing sent), a busy broker (broker code Unavailable), or CNT_PANIC: then it still holds the lease,
    /// to commit again or release.
    Status commit(std::span<const cnt_output> outputs, const cnt_coverage* coverage = nullptr,
                  const cnt_cost* cost = nullptr) noexcept {
        if (!lease_) return Status{CNT_INVALID_ARGUMENT, 0, "the lease already ended"};
        Status s = Status::last(cnt_work_lease_commit(lease_, outputs.data(), outputs.size(), coverage, cost));
        bool kept = s.code == CNT_INVALID_ARGUMENT || s.code == CNT_PANIC || (s.code == CNT_BROKER && s.broker_code == 5);
        if (!kept) lease_ = nullptr;
        return s;
    }
    /// Gives up the job. The WorkLease is empty afterward either way.
    Status abort(const std::string& reason) noexcept {
        if (!lease_) return Status{CNT_INVALID_ARGUMENT, 0, "the lease already ended"};
        return Status::last(cnt_work_lease_abort(std::exchange(lease_, nullptr), reason.c_str()));
    }
    /// Gives the lease back without committing (after a preemption, or early). No-op when empty.
    void release() noexcept {
        if (lease_) cnt_work_lease_release(std::exchange(lease_, nullptr));
    }

private:
    cnt_work_lease* lease_ = nullptr;
};

namespace detail {
/// A move-only owner of a C handle, freed by `Free` when destroyed.
template <typename T, void (*Free)(T*)>
class Owned {
public:
    Owned() noexcept = default;
    explicit Owned(T* p) noexcept : p_(p) {}
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
    Owned(Owned&& other) noexcept : p_(std::exchange(other.p_, nullptr)) {}
    Owned& operator=(Owned&& other) noexcept {
        if (this != &other) {
            reset();
            p_ = std::exchange(other.p_, nullptr);
        }
        return *this;
    }
    ~Owned() { reset(); }
    T* get() const noexcept { return p_; }
    T* release() noexcept { return std::exchange(p_, nullptr); }
    void reset(T* p = nullptr) noexcept {
        if (p_) Free(p_);
        p_ = p;
    }
    explicit operator bool() const noexcept { return p_ != nullptr; }

private:
    T* p_ = nullptr;
};
}  // namespace detail

/// A Tier 0 slot claimed for writing (from Client::claim_slot): fill bytes(), then publish. Destroying it
/// unpublished gives the slot back.
class WriteSlot {
public:
    WriteSlot() noexcept = default;
    explicit WriteSlot(cnt_write_slot* slot) noexcept : slot_(slot) {}

    /// The slot's bytes to fill, 16 KB aligned. Empty when empty.
    std::span<std::byte> bytes() noexcept {
        if (!slot_) return {};
        return {reinterpret_cast<std::byte*>(cnt_write_slot_data(slot_.get())), cnt_write_slot_capacity(slot_.get())};
    }
    /// Whether this holds a claimed slot.
    explicit operator bool() const noexcept { return static_cast<bool>(slot_); }

    /// Publishes the first len bytes and writes where they are to at (for Client::ingest or a commit's output). The
    /// WriteSlot is empty afterward either way.
    Status publish(std::size_t len, cnt_location& at) noexcept {
        if (!slot_) return Status{CNT_INVALID_ARGUMENT, 0, "no slot"};
        return Status::last(cnt_write_slot_publish(slot_.release(), len, &at));
    }

private:
    detail::Owned<cnt_write_slot, cnt_write_slot_abandon> slot_;
};

/// A list of payloads, borrowed from what holds it (a QueryResult or a Job): valid while that lives.
class ItemsView {
public:
    ItemsView() noexcept = default;
    explicit ItemsView(const cnt_items* items) noexcept : items_(items) {}

    std::size_t size() const noexcept { return cnt_items_count(items_); }
    bool empty() const noexcept { return size() == 0; }
    /// Payload i: its header and where its bytes are, pointing into the list. An empty item past the end.
    cnt_item operator[](std::size_t i) const noexcept {
        cnt_item item{};
        if (cnt_items_at(items_, i, &item) != CNT_OK) return cnt_item{};
        return item;
    }
    /// Payload i's header bytes.
    std::span<const std::uint8_t> header(std::size_t i) const noexcept {
        cnt_item item = (*this)[i];
        return {item.header, item.header_len};
    }

private:
    const cnt_items* items_ = nullptr;
};

/// A list of payloads this owns (a promise's part, a subscription event, an event's outputs).
class Items {
public:
    Items() noexcept = default;
    explicit Items(cnt_items* items) noexcept : items_(items) {}
    ItemsView view() const noexcept { return ItemsView(items_.get()); }
    std::size_t size() const noexcept { return view().size(); }
    cnt_item operator[](std::size_t i) const noexcept { return view()[i]; }
    explicit operator bool() const noexcept { return static_cast<bool>(items_); }

private:
    detail::Owned<cnt_items, cnt_items_free> items_;
};

/// What a query found (from Client::query).
class QueryResult {
public:
    QueryResult() noexcept = default;
    explicit QueryResult(cnt_query_result* r) noexcept : r_(r) {}

    /// The payloads, in time order (valid while this lives).
    ItemsView items() const noexcept { return ItemsView(cnt_query_items(r_.get())); }
    /// Whether the items cover the whole range.
    bool complete() const noexcept { return cnt_query_complete(r_.get()); }
    /// Whether the page was cut short of the range.
    bool truncated() const noexcept { return cnt_query_truncated(r_.get()); }
    /// The intervals with nothing (UTC ns, half open).
    std::vector<std::pair<std::uint64_t, std::uint64_t>> missing() const {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
        for (std::size_t i = 0, n = cnt_query_missing_count(r_.get()); i < n; ++i) {
            std::uint64_t a = 0, b = 0;
            if (cnt_query_missing_at(r_.get(), i, &a, &b) == CNT_OK) out.emplace_back(a, b);
        }
        return out;
    }
    /// The quote for what's missing, when there is one.
    bool quote(cnt_promise_quote& out) const noexcept { return cnt_query_quote(r_.get(), &out) == CNT_OK; }

private:
    detail::Owned<cnt_query_result, cnt_query_result_free> r_;
};

/// A committed promise (from Client::commit_promise): its outputs as they come. Destroying it stops routing them
/// here; it doesn't cancel the promise. Use it while its Client lives.
class Promise {
public:
    Promise() noexcept = default;
    explicit Promise(cnt_promise* p) noexcept : p_(p) {}
    explicit operator bool() const noexcept { return static_cast<bool>(p_); }

    /// Waits up to timeout_ms for the next part. CNT_CLOSED once the promise is over; CNT_BROKER with the reason if it
    /// failed.
    Status next_part(std::uint32_t timeout_ms, Items& out, bool& final_part) noexcept {
        cnt_items* items = nullptr;
        std::uint8_t last = 0;
        cnt_status s = cnt_promise_next_part(p_.get(), timeout_ms, &items, &last);
        if (s == CNT_OK) {
            out = Items(items);
            final_part = last != 0;
        }
        return Status::last(s);
    }
    /// Gathers every part through the final one, each waited for up to timeout_ms.
    Status fulfilled(std::uint32_t timeout_ms, std::vector<Items>& parts) {
        for (;;) {
            Items part;
            bool last = false;
            Status s = next_part(timeout_ms, part, last);
            if (!s) return s;
            parts.push_back(std::move(part));
            if (last) return {};
        }
    }

private:
    detail::Owned<cnt_promise, cnt_promise_free> p_;
};

/// Something a Subscription delivered.
struct SubEvent {
    /// CNT_SUB_EVENT, CNT_SUB_CAUGHT_UP, or CNT_SUB_UNDECODABLE.
    std::uint8_t type = 0;
    std::uint64_t position = 0;
    /// For CNT_SUB_EVENT, the payload (one item).
    Items item;
    /// For CNT_SUB_UNDECODABLE, why.
    std::string reason;
};

/// A live subscription (from Client::subscribe). Destroying it ends it. Use it while its Client lives.
class Subscription {
public:
    Subscription() noexcept = default;
    explicit Subscription(cnt_subscription* s) noexcept : s_(s) {}
    explicit operator bool() const noexcept { return static_cast<bool>(s_); }

    /// Waits up to timeout_ms for the next delivery. CNT_TIMEOUT if none came.
    Status next(std::uint32_t timeout_ms, SubEvent& out) {
        cnt_sub_event e{};
        cnt_status s = cnt_subscription_next(s_.get(), timeout_ms, &e);
        if (s == CNT_OK) out = SubEvent{e.type, e.position, Items(e.item), e.reason};
        return Status::last(s);
    }
    /// Says everything up to position is handled (a durable subscription resumes after it).
    Status ack(std::uint64_t position) noexcept { return Status::last(cnt_subscription_ack(s_.get(), position)); }

private:
    detail::Owned<cnt_subscription, cnt_subscription_free> s_;
};

/// A job the broker dispatched (from an Event of type CNT_EVENT_DISPATCH_JOB). Take its output lease with
/// Client::take_lease, read inputs(), and commit on the lease (or abort it to turn the job down).
class Job {
public:
    Job() noexcept = default;
    explicit Job(cnt_job* j) noexcept : j_(j) {}
    explicit operator bool() const noexcept { return static_cast<bool>(j_); }
    cnt_job* handle() const noexcept { return j_.get(); }

    /// What it asks for.
    cnt_job_info info() const noexcept {
        cnt_job_info out{};
        cnt_job_describe(j_.get(), &out);
        return out;
    }
    /// Its inputs, resolved and pinned by the broker (valid while this lives).
    ItemsView inputs() const noexcept { return ItemsView(cnt_job_inputs(j_.get())); }
    /// The promises it serves.
    std::vector<std::array<std::uint8_t, 16>> promises() const {
        std::vector<std::array<std::uint8_t, 16>> out(info().n_promises);
        for (std::size_t i = 0; i < out.size(); ++i) cnt_job_promise_at(j_.get(), i, out[i].data());
        return out;
    }

private:
    detail::Owned<cnt_job, cnt_job_free> j_;
};

/// Something the broker pushed (from Client::next_event), owning what came with it: a dispatched job, or a promise's
/// outputs.
struct Event {
    cnt_event raw{};
    /// For CNT_EVENT_DISPATCH_JOB.
    Job job;
    /// For CNT_EVENT_PROMISE_FULFILLED.
    Items items;
};

#if defined(__cpp_exceptions)
/// Builds a header, or throws Error.
inline std::vector<std::uint8_t> build_header(const cnt_header_spec& spec) {
    std::vector<std::uint8_t> out;
    detail::check(build_header(spec, out));
    return out;
}
/// Reads a header, or throws Error.
inline cnt_header_info read_header(std::span<const std::uint8_t> header) {
    cnt_header_info out{};
    detail::check(read_header(header, out));
    return out;
}
#endif

/// How to connect.
struct ConnectOptions {
    std::string client_name = "cpp";
    std::uint32_t heartbeat_interval_ms = 5000;
    std::uint32_t request_timeout_ms = 10000;
};

/// A stream to register. format and layer are the schema's PayloadFormat and DomainLayer values.
struct StreamSpec {
    std::string name;
    std::uint8_t kind = CNT_DERIVED;
    std::uint8_t format = 0;  // RawBytes
    std::uint8_t layer = 0;   // Layer1Raw
    std::string shape_name;
    std::uint32_t shape_version = 1;
    float nominal_rate_hz = 0.0f;
    std::uint8_t storage = CNT_SLAB;
    std::uint16_t slab_class = 0;
    std::uint32_t ring_stride_bytes = 0;
    float ring_seconds = 0.0f;
    bool has_anchor = false;
    std::array<std::uint8_t, 16> anchor_boot_id{};
    std::uint64_t anchor_monotonic_ns = 0;
    std::uint64_t anchor_utc_ns = 0;
};

/// An arena the broker handed the client.
struct ArenaInfo {
    std::string name;
    std::uint32_t generation = 0;
    std::uint64_t size_bytes = 0;
    const std::byte* base = nullptr;
};

/// A connection to the broker.
class Client {
public:
    Client() noexcept = default;
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&& other) noexcept : client_(std::exchange(other.client_, nullptr)) {}
    Client& operator=(Client&& other) noexcept {
        if (this != &other) {
            reset();
            client_ = std::exchange(other.client_, nullptr);
        }
        return *this;
    }
    /// Says Goodbye (if not already closed) and frees the connection.
    ~Client() { reset(); }

    /// Whether this holds a connection.
    explicit operator bool() const noexcept { return client_ != nullptr; }
    /// The C handle, for calls this wrapper doesn't cover yet.
    cnt_client* handle() const noexcept { return client_; }

    /// Connects to the broker's socket and maps the arenas it hands over.
    static Status connect(const std::string& socket, const ConnectOptions& options, Client& out) noexcept {
        if (cnt_abi_version() != abi_version) {
            return Status{CNT_INVALID_ARGUMENT, 0, "libcontinuum's ABI version doesn't match continuum.hpp"};
        }
        cnt_connect_options o = cnt_connect_options_default();
        o.client_name = options.client_name.c_str();
        o.heartbeat_interval_ms = options.heartbeat_interval_ms;
        o.request_timeout_ms = options.request_timeout_ms;
        cnt_client* c = nullptr;
        cnt_status s = cnt_connect(socket.c_str(), &o, &c);
        if (s == CNT_OK) out = Client(c);
        return Status::last(s);
    }

    /// Sends a heartbeat; writes the broker's clock (UTC nanoseconds) to broker_now_ns.
    Status heartbeat(std::uint64_t& broker_now_ns) noexcept {
        return Status::last(cnt_heartbeat(client_, &broker_now_ns));
    }

    /// Registers a stream; writes its id and standing lease (if any) to out.
    Status register_stream(const StreamSpec& spec, cnt_stream_registered& out) noexcept {
        cnt_stream_spec c{};
        c.name = spec.name.c_str();
        c.kind = spec.kind;
        c.format = spec.format;
        c.layer = spec.layer;
        c.shape_name = spec.shape_name.c_str();
        c.shape_version = spec.shape_version;
        c.nominal_rate_hz = spec.nominal_rate_hz;
        c.storage = spec.storage;
        c.slab_class = spec.slab_class;
        c.ring_stride_bytes = spec.ring_stride_bytes;
        c.ring_seconds = spec.ring_seconds;
        c.has_anchor = spec.has_anchor ? 1 : 0;
        for (std::size_t i = 0; i < 16; ++i) c.anchor_boot_id[i] = spec.anchor_boot_id[i];
        c.anchor_monotonic_ns = spec.anchor_monotonic_ns;
        c.anchor_utc_ns = spec.anchor_utc_ns;
        return Status::last(cnt_register_stream(client_, &c, &out));
    }

    /// Pins a published slot of arena `arena` for ttl_ms milliseconds.
    Status lease(std::size_t arena, cnt_slot_ref slot, std::uint32_t ttl_ms, Lease& out) noexcept {
        cnt_lease* l = nullptr;
        cnt_status s = cnt_lease_slot(client_, arena, slot, ttl_ms, &l);
        if (s == CNT_OK) out = Lease(l);
        return Status::last(s);
    }

    /// Registers this session as a worker; results gets one answer per capability, in order, and reasons (if
    /// non-null) the refusal reasons, one per line.
    Status register_worker(const std::string& worker_id, std::span<const cnt_capability> capabilities,
                           std::vector<cnt_capability_result>& results, std::string* reasons = nullptr,
                           const std::string& version_hash = "", std::int16_t device_index = -1,
                           bool keeps_copies = false) {
        results.assign(capabilities.size(), cnt_capability_result{});
        std::array<char, 1024> buf{};
        cnt_status s = cnt_register_worker(client_, worker_id.c_str(), version_hash.c_str(), device_index, keeps_copies ? 1 : 0,
                                           capabilities.data(), capabilities.size(), results.data(),
                                           reasons ? buf.data() : nullptr, reasons ? buf.size() : 0);
        if (s == CNT_OK && reasons) *reasons = buf.data();
        return Status::last(s);
    }

    /// Asks for a work lease. CNT_DENIED with retry_after_ms set when the broker says not now.
    Status quote(const cnt_quote_spec& spec, WorkLease& out, std::uint32_t* retry_after_ms = nullptr) noexcept {
        cnt_work_lease* l = nullptr;
        cnt_status s = cnt_quote(client_, &spec, &l, retry_after_ms);
        if (s == CNT_OK) out = WorkLease(l);
        return Status::last(s);
    }

    /// Tells the broker a payload is written (one way; a refusal comes back as an event).
    Status ingest(std::span<const std::uint8_t> header, const cnt_location* at = nullptr) noexcept {
        return Status::last(cnt_ingest(client_, header.data(), header.size(), at));
    }

    /// Sets the voice turn (the schema's VoiceTurnState value). For the voice service.
    Status set_system_state(std::uint8_t voice_turn) noexcept {
        return Status::last(cnt_set_system_state(client_, voice_turn));
    }

    /// Waits up to timeout_ms for the next message the broker pushes. CNT_TIMEOUT if none came. A dispatched job or
    /// a promise's outputs are out's to own.
    Status next_event(std::uint32_t timeout_ms, Event& out) noexcept {
        cnt_event e{};
        cnt_status s = cnt_next_event(client_, timeout_ms, &e);
        if (s == CNT_OK) {
            out.job = Job(std::exchange(e.job, nullptr));
            out.items = Items(std::exchange(e.items, nullptr));
            out.raw = e;
        }
        return Status::last(s);
    }

    /// Claims a free slot of slab_class in arena `arena` (Welcome order) for writing, tagged with owner.
    Status claim_slot(std::size_t arena, std::uint16_t slab_class, std::uint64_t owner, WriteSlot& out) noexcept {
        cnt_write_slot* w = nullptr;
        cnt_status s = cnt_claim_slot(client_, arena, slab_class, owner, &w);
        if (s == CNT_OK) out = WriteSlot(w);
        return Status::last(s);
    }

    /// Asks what a stream has over a range (and, in a quoting mode, a quote for what's missing).
    Status query(const cnt_query_spec& spec, QueryResult& out) noexcept {
        cnt_query_result* r = nullptr;
        cnt_status s = cnt_query(client_, &spec, &r);
        if (s == CNT_OK) out = QueryResult(r);
        return Status::last(s);
    }

    /// Commits a quoted promise (option 0 is the broker's pick); its outputs come to out. Writes its PromiseLife to
    /// life if non-null.
    Status commit_promise(const std::array<std::uint8_t, 16>& promise_id, std::uint32_t option, std::uint8_t band,
                          Promise& out, std::uint8_t* life = nullptr) noexcept {
        cnt_promise* p = nullptr;
        cnt_status s = cnt_commit_promise(client_, promise_id.data(), option, band, life, &p);
        if (s == CNT_OK) out = Promise(p);
        return Status::last(s);
    }

    /// Calls a promise off; writes its PromiseLife to life if non-null.
    Status cancel_promise(const std::array<std::uint8_t, 16>& promise_id, std::uint8_t* life = nullptr) noexcept {
        return Status::last(cnt_cancel_promise(client_, promise_id.data(), life));
    }

    /// Follows streams from start (CNT_START_LIVE, _TIME with start_ns, or _SAVED with durable_name).
    Status subscribe(const std::vector<std::string>& streams, std::uint8_t start, Subscription& out,
                     std::uint64_t start_ns = 0, const std::string& durable_name = "", bool pinned = false) {
        std::vector<const char*> names;
        for (const std::string& n : streams) names.push_back(n.c_str());
        cnt_subscribe_spec spec{};
        spec.streams = names.data();
        spec.n_streams = names.size();
        spec.start = start;
        spec.start_ns = start_ns;
        spec.durable_name = durable_name.empty() ? nullptr : durable_name.c_str();
        spec.pinned = pinned ? 1 : 0;
        cnt_subscription* sub = nullptr;
        cnt_status s = cnt_subscribe(client_, &spec, &sub);
        if (s == CNT_OK) out = Subscription(sub);
        return Status::last(s);
    }

    /// Takes a dispatched job's output lease (once per job).
    Status take_lease(Job& job, WorkLease& out) noexcept {
        cnt_work_lease* l = nullptr;
        cnt_status s = cnt_job_take_lease(client_, job.handle(), &l);
        if (s == CNT_OK) out = WorkLease(l);
        return Status::last(s);
    }

    /// Says Goodbye now; the broker drops the session's pins and leases.
    Status close(const std::string& reason = "closed") noexcept {
        return Status::last(cnt_client_close(client_, reason.c_str()));
    }

    /// The session id (a UUIDv7, 16 bytes).
    std::array<std::uint8_t, 16> session_id() const noexcept {
        std::array<std::uint8_t, 16> id{};
        cnt_session_id(client_, id.data());
        return id;
    }

    /// The arenas the broker handed over, in Welcome order.
    std::vector<ArenaInfo> arenas() const {
        std::vector<ArenaInfo> out;
        std::size_t n = cnt_arena_count(client_);
        for (std::size_t i = 0; i < n; ++i) {
            cnt_arena_info info{};
            if (cnt_arena_info_at(client_, i, &info) != CNT_OK) continue;
            out.push_back({info.name, info.generation, info.size_bytes, reinterpret_cast<const std::byte*>(info.base)});
        }
        return out;
    }

#if defined(__cpp_exceptions)
    /// Connects, or throws Error.
    static Client connect(const std::string& socket, const ConnectOptions& options = {}) {
        Client c;
        detail::check(connect(socket, options, c));
        return c;
    }
    /// Sends a heartbeat and returns the broker's clock, or throws Error.
    std::uint64_t heartbeat() {
        std::uint64_t now = 0;
        detail::check(heartbeat(now));
        return now;
    }
    /// Registers a stream, or throws Error.
    cnt_stream_registered register_stream(const StreamSpec& spec) {
        cnt_stream_registered out{};
        detail::check(register_stream(spec, out));
        return out;
    }
    /// Asks for a work lease, or throws Error (CNT_DENIED included).
    WorkLease quote(const cnt_quote_spec& spec) {
        WorkLease l;
        detail::check(quote(spec, l));
        return l;
    }
    /// Claims a slot for writing, or throws Error.
    WriteSlot claim_slot(std::size_t arena, std::uint16_t slab_class, std::uint64_t owner = 0) {
        WriteSlot w;
        detail::check(claim_slot(arena, slab_class, owner, w));
        return w;
    }
    /// Queries, or throws Error.
    QueryResult query(const cnt_query_spec& spec) {
        QueryResult r;
        detail::check(query(spec, r));
        return r;
    }
    /// Commits a quoted promise, or throws Error.
    Promise commit_promise(const std::array<std::uint8_t, 16>& promise_id, std::uint32_t option = 0,
                           std::uint8_t band = 2) {
        Promise p;
        detail::check(commit_promise(promise_id, option, band, p));
        return p;
    }
    /// Takes a dispatched job's output lease, or throws Error.
    WorkLease take_lease(Job& job) {
        WorkLease l;
        detail::check(take_lease(job, l));
        return l;
    }
    /// Pins a published slot, or throws Error.
    Lease lease(std::size_t arena, cnt_slot_ref slot, std::uint32_t ttl_ms = 5000) {
        Lease l;
        detail::check(lease(arena, slot, ttl_ms, l));
        return l;
    }
#endif

private:
    explicit Client(cnt_client* c) noexcept : client_(c) {}
    void reset() noexcept {
        if (client_) {
            cnt_client_close(client_, "closed");  // CNT_CLOSED if it already was; nothing to do then
            cnt_client_free(std::exchange(client_, nullptr));
        }
    }
    cnt_client* client_ = nullptr;
};

}  // namespace continuum
