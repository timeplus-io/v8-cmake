/// proton: starts
/// Provide a minimal std::atomic_ref<T> shim when the standard library
/// does not offer it (older libstdc++/libc++ or disabled feature macros).
/// Only load() and store() are implemented as needed by V8 sources.
#pragma once

#include <atomic>
#include <type_traits>
#include <cstdint>

namespace std {

#if !defined(__cpp_lib_atomic_ref) || (__cpp_lib_atomic_ref + 0) < 201806L
template <class T>
class atomic_ref {
  static_assert(std::is_trivially_copyable<T>::value,
                "atomic_ref requires trivially copyable T");

 public:
  using value_type = T;

  explicit atomic_ref(T& obj) noexcept : ptr_(&obj) {}
  atomic_ref(T&&) = delete;
  atomic_ref(const atomic_ref&) = default;
  atomic_ref& operator=(const atomic_ref&) = default;

  void store(T desired,
             std::memory_order order = std::memory_order_seq_cst) const
      noexcept {
    __atomic_store(ptr_, &desired, order_to_gcc(order));
  }

  T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    T out;
    __atomic_load(ptr_, &out, order_to_gcc(order));
    return out;
  }

 private:
  static int order_to_gcc(std::memory_order mo) noexcept {
    switch (mo) {
      case std::memory_order_relaxed:
        return __ATOMIC_RELAXED;
      case std::memory_order_consume:
        return __ATOMIC_CONSUME;
      case std::memory_order_acquire:
        return __ATOMIC_ACQUIRE;
      case std::memory_order_release:
        return __ATOMIC_RELEASE;
      case std::memory_order_acq_rel:
        return __ATOMIC_ACQ_REL;
      case std::memory_order_seq_cst:
      default:
        return __ATOMIC_SEQ_CST;
    }
  }

  T* ptr_;
};
#endif // !__cpp_lib_atomic_ref

}  // namespace std
/// proton: ends

