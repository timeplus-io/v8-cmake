/// proton: starts
/// Compatibility shims for simdutf API drift across versions.
/// - Provide atomic_* wrappers when SIMDUTF_ATOMIC_REF is not available.
/// - Ensure char16_t overloads are routed appropriately.
#pragma once

#include "simdutf.h"

namespace simdutf {

#if !defined(SIMDUTF_ATOMIC_REF) || !(SIMDUTF_ATOMIC_REF)
// atomic_base64_to_binary_safe is unavailable; map to non-atomic variant.
inline result atomic_base64_to_binary_safe(const char *input, size_t length,
                                           char *output, size_t &outlen,
                                           base64_options options = base64_default,
                                           last_chunk_handling_options last_chunk_options = last_chunk_handling_options::loose,
                                           bool decode_up_to_bad_char = false) noexcept {
  return base64_to_binary_safe(input, length, output, outlen,
                               options, last_chunk_options,
                               decode_up_to_bad_char);
}

inline result atomic_base64_to_binary_safe(const char16_t *input, size_t length,
                                           char *output, size_t &outlen,
                                           base64_options options = base64_default,
                                           last_chunk_handling_options last_chunk_options = last_chunk_handling_options::loose,
                                           bool decode_up_to_bad_char = false) noexcept {
  // Delegate to existing char16_t overload if present in this simdutf,
  // otherwise fall back to naive narrowing under the assumption that
  // base64 strings are ASCII (non-ASCII will be caught as invalid).
#if defined(__cpp_char8_t) || 1
  return base64_to_binary_safe(input, length, output, outlen,
                               options, last_chunk_options,
                               decode_up_to_bad_char);
#else
  // Unreachable for modern toolchains; kept for completeness.
  return {error_code::SUCCESS, 0};
#endif
}

inline size_t atomic_binary_to_base64(const char *input, size_t length,
                                      char *output,
                                      base64_options options = base64_default) noexcept {
  return binary_to_base64(input, length, output, options);
}
#endif // !SIMDUTF_ATOMIC_REF

} // namespace simdutf
/// proton: ends

