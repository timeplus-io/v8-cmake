// Minimal shim for systems where kernel UAPI headers are not available
// in the configured sysroot. Provides AT_HWCAP and AT_HWCAP2 constants
// used by V8 to query getauxval()/auxv entries.
#ifndef V8_SHIM_LINUX_AUXVEC_H_
#define V8_SHIM_LINUX_AUXVEC_H_

#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif

#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif

#endif  // V8_SHIM_LINUX_AUXVEC_H_

