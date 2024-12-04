cmake_minimum_required(VERSION 3.20)
project(V8 LANGUAGES CXX C ASM)

# V8 CMake Build Configuration
#
# Cross-compilation support:
# - For native builds, tools (bytecode_builtins_list_generator, torque, mksnapshot) 
#   are built from source
# - For cross-compilation, pre-built versions of these tools must be provided in 
#   CMAKE_CURRENT_SOURCE_DIR
#
# Required pre-built tools for cross-compilation:
# - ${CMAKE_CURRENT_SOURCE_DIR}/bytecode_builtins_list_generator
# - ${CMAKE_CURRENT_SOURCE_DIR}/torque  
# - ${CMAKE_CURRENT_SOURCE_DIR}/mksnapshot
if (CMAKE_HOST_SYSTEM_NAME STREQUAL CMAKE_SYSTEM_NAME
  AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL CMAKE_SYSTEM_PROCESSOR)
  set(V8_CROSSCOMPILE OFF)
else()
  set(V8_CROSSCOMPILE ON)
endif()

option(V8CMAKE_FIND_PYTHON "Enable finding python" ON)

add_definitions("-DV8_ENABLE_WEBASSEMBLY -DV8_ADVANCED_BIGINT_ALGORITHMS -DV8_ENABLE_WASM_SIMD256_REVEC")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (V8CMAKE_FIND_PYTHON)
  find_package(Python 3 REQUIRED QUIET COMPONENTS Interpreter)
endif()
find_package(Threads QUIET)

include(CMakeDependentOption)
include(GNUInstallDirs)

include(CheckPythonModuleExists)

check_python_module_exists(PYTHON_HAVE_MARKUPSAFE markupsafe)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if (NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 90)
endif()

set(is-freebsd $<PLATFORM_ID:FreeBSD>)
set(is-darwin $<PLATFORM_ID:Darwin>)
set(is-linux $<PLATFORM_ID:Linux>)
set(is-win $<PLATFORM_ID:Windows>)
set(is-aix $<PLATFORM_ID:AIX>)
string(CONCAT is-posix $<OR:
  ${is-freebsd},
  ${is-darwin},
  ${is-linux},
  ${is-aix}
>)

set(is-msvc $<CXX_COMPILER_ID:MSVC>)

# FIXME obviously
set(is-arm 0)
set(is-arm64 0)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(is-arm64 1)
endif ()
set(is-ia32 0)
set(is-mips 0)
set(is-mips64 0)
set(is-ppc 0)

set(is-s390 $<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},S390>)
string(CONCAT is-x64 $<OR:
  $<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>,
  $<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>
>)

option(V8_ENABLE_CONCURRENT_MARKING "Enable concurrent marking" ON)
option(V8_ENABLE_I18N "Enable Internationalization support")

set(
  v8_defines
  $<${is-darwin}:V8_HAVE_TARGET_OS>
  $<${is-darwin}:V8_TARGET_OS_MACOS>
  $<${is-linux}:V8_HAVE_TARGET_OS>
  $<${is-linux}:V8_TARGET_OS_LINUX>
  $<${is-win}:V8_HAVE_TARGET_OS>
  $<${is-win}:V8_TARGET_OS_WIN>
  $<${is-x64}:V8_TARGET_ARCH_X64>
  $<${is-arm64}:V8_TARGET_ARCH_ARM64>
  $<${is-win}:NOMINMAX>
  $<$<AND:${is-win},${is-x64}>:V8_OS_WIN_X64>
  $<$<BOOL:${V8_ENABLE_CONCURRENT_MARKING}>:V8_CONCURRENT_MARKING>
  $<${is-win}:V8_OS_WIN32>
)

set(disable-exceptions
  $<$<CXX_COMPILER_ID:MSVC>:/EHs-c->
  $<$<CXX_COMPILER_ID:AppleClang>:-fno-exceptions>
  $<$<CXX_COMPILER_ID:Clang>:-fno-exceptions -Wno-invalid-offsetof -Wno-deprecated-non-prototype>
  $<$<CXX_COMPILER_ID:GNU>:-fno-exceptions>
)

set(enable-exceptions
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
  $<$<CXX_COMPILER_ID:AppleClang>:-fexceptions>
  $<$<CXX_COMPILER_ID:Clang>:-fexceptions>
  $<$<CXX_COMPILER_ID:GNU>:-fexceptions>
)

set(enable-fpic
  $<$<CXX_COMPILER_ID:AppleClang>:-fPIC>
  $<$<CXX_COMPILER_ID:Clang>:-fPIC>
  $<$<CXX_COMPILER_ID:GNU>:-fPIC>
)

# more granular library support
add_library(v8-i18n-support OBJECT)
set_property(TARGET v8-i18n-support PROPERTY EXCLUDE_FROM_ALL ON)

list(APPEND i18n-sources
  v8/src/builtins/builtins-intl.cc
  v8/src/objects/intl-objects.cc
  v8/src/objects/js-break-iterator.cc
  v8/src/objects/js-collator.cc
  v8/src/objects/js-date-time-format.cc
  v8/src/objects/js-duration-format.cc
  v8/src/objects/js-display-names.cc
  v8/src/objects/js-list-format.cc
  v8/src/objects/js-locale.cc
  v8/src/objects/js-number-format.cc
  v8/src/objects/js-plural-rules.cc
  v8/src/objects/js-relative-time-format.cc
  v8/src/objects/js-segment-iterator.cc
  v8/src/objects/js-segmenter.cc
  v8/src/objects/js-segments.cc
  v8/src/runtime/runtime-intl.cc
  v8/src/strings/char-predicates.cc
)

target_sources(v8-i18n-support PRIVATE ${i18n-sources})
target_compile_definitions(v8-i18n-support PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8-i18n-support PRIVATE ${disable-exceptions})
target_link_libraries(v8-i18n-support PRIVATE v8_torque_generated)

target_include_directories(v8-i18n-support
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/src/objects
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
)

file(GLOB api-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/api/*.cc)
file(GLOB asmjs-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/asmjs/*.cc)
file(GLOB ast-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/ast/*.cc)
file(GLOB builtin-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/builtins/*.cc)
file(GLOB baseline-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/baseline/*.cc)
file(GLOB bigint-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/bigint/*.cc)
file(GLOB codegen-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/codegen/*.cc)
file(GLOB debug-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/debug/*.cc)
file(GLOB deoptimizer-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/deoptimizer/*.cc)
file(GLOB diagnostic-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/diagnostics/*.cc)
file(GLOB execution-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/execution/*.cc)
file(GLOB extensions-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/extensions/*.cc)
file(GLOB handles-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/handles/*.cc)
file(GLOB interpreter-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/interpreter/*.cc)
file(GLOB json-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/json/*.cc)
file(GLOB logging-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/logging/*.cc)
file(GLOB numbers-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/base/numbers/*.cc v8/src/numbers/*.cc)
file(GLOB object-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/objects/*.cc)
file(GLOB parsing-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/parsing/*.cc)
file(GLOB profiler-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/profiler/*.cc)
file(GLOB regexp-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/regexp/*.cc v8/src/regexp/experimental/*.cc)
file(GLOB runtime-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/runtime/*.cc)
file(GLOB snapshot-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/snapshot/*.cc)
file(GLOB strings-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/strings/*.cc)
file(GLOB utils-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/utils/*.cc)
file(GLOB wasm-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/wasm/*.cc)
file(GLOB wasm-baseline-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/wasm/baseline/*.cc)
file(GLOB zone-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/src/zone/*.cc)
file(GLOB zlib-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS v8/third_party/zlib/*.c v8/third_party/zlib/google/compression_utils_portable.cc)

file(GLOB heap-sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
  v8/src/heap/*.cc
  v8/src/heap/base/*.cc
  v8/src/heap/cppgc-js/*.cc
  v8/src/heap/cppgc/*.cc
  v8/src/heap/third-party/*.cc
)

# Caged heap feature is x64/arm64-only at the moment.
list(REMOVE_ITEM heap-sources
  v8/src/heap/conservative-stack-visitor.cc
  v8/src/heap/cppgc/caged-heap-local-data.cc
  v8/src/heap/cppgc/caged-heap.cc
)

list(APPEND heap-sources
  $<${is-arm64}:v8/src/heap/base/asm/arm64/push_registers_asm.cc>
  $<${is-arm}:v8/src/heap/base/asm/arm/push_registers_asm.cc>
  $<${is-ia32}:v8/src/heap/base/asm/ia32/push_registers_asm.cc>
  $<${is-mips64}:v8/src/heap/base/asm/mips64/push_registers_asm.cc>
  $<${is-mips}:v8/src/heap/base/asm/mips/push_registers_asm.cc>
  $<${is-ppc}:v8/src/heap/base/asm/ppc/push_registers_asm.cc>
  $<${is-s390}:v8/src/heap/base/asm/s390/push_registers_asm.cc>
  $<${is-x64}:v8/src/heap/base/asm/x64/push_registers_asm.cc>
)

list(REMOVE_ITEM builtin-sources
  v8/src/builtins/builtins-intl-gen.cc
  v8/src/builtins/builtins-intl.cc
)

list(REMOVE_ITEM diagnostic-sources v8/src/diagnostics/unwinding-info-win64.cc)
list(REMOVE_ITEM diagnostic-sources v8/src/diagnostics/etw-jit-win.cc)

list(REMOVE_ITEM object-sources ${i18n-sources})
list(REMOVE_ITEM regexp-sources v8/src/regexp/gen-regexp-special-case.cc)
list(REMOVE_ITEM runtime-sources v8/src/runtime/runtime-intl.cc)
list(REMOVE_ITEM snapshot-sources
  v8/src/snapshot/mksnapshot.cc
  v8/src/snapshot/snapshot-empty.cc
  v8/src/snapshot/snapshot-external.cc
)
list(REMOVE_ITEM strings-sources v8/src/strings/char-predicates.cc)

list(APPEND snapshot-sources v8/src/snapshot/embedded/embedded-data.cc)
list(APPEND wasm-sources ${wasm-baseline-sources})

add_library(v8_base_without_compiler STATIC
  $<${is-posix}:v8/src/trap-handler/handler-inside-posix.cc>
  $<${is-posix}:v8/src/trap-handler/handler-outside-posix.cc>
  $<$<AND:${is-x64},${is-win}>:v8/src/diagnostics/unwinding-info-win64.cc>
  $<$<AND:${is-x64},${is-win}>:v8/src/trap-handler/handler-inside-win.cc>
  $<$<AND:${is-x64},${is-win}>:v8/src/trap-handler/handler-outside-win.cc>
  $<${is-x64}:v8/src/codegen/x64/assembler-x64.cc>
  $<${is-x64}:v8/src/codegen/x64/cpu-x64.cc>
  $<${is-x64}:v8/src/codegen/x64/macro-assembler-x64.cc>
  $<${is-x64}:v8/src/codegen/shared-ia32-x64/macro-assembler-shared-ia32-x64.cc>
  $<${is-x64}:v8/src/compiler/backend/x64/code-generator-x64.cc>
  $<${is-x64}:v8/src/compiler/backend/x64/instruction-scheduler-x64.cc>
  $<${is-x64}:v8/src/compiler/backend/x64/instruction-selector-x64.cc>
  $<${is-x64}:v8/src/compiler/backend/x64/unwinding-info-writer-x64.cc>
  $<${is-x64}:v8/src/deoptimizer/x64/deoptimizer-x64.cc>
  $<${is-x64}:v8/src/diagnostics/x64/disasm-x64.cc>
  $<${is-x64}:v8/src/diagnostics/x64/eh-frame-x64.cc>
  $<${is-x64}:v8/src/execution/x64/frame-constants-x64.cc>
  $<${is-x64}:v8/src/regexp/x64/regexp-macro-assembler-x64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/assembler-arm64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/cpu-arm64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/decoder-arm64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/macro-assembler-arm64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/instructions-arm64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/instructions-arm64-constants.cc>
  $<${is-arm64}:v8/src/codegen/arm64/register-arm64.cc>
  $<${is-arm64}:v8/src/codegen/arm64/utils-arm64.cc>
  $<${is-arm64}:v8/src/compiler/backend/arm64/code-generator-arm64.cc>
  $<${is-arm64}:v8/src/compiler/backend/arm64/instruction-scheduler-arm64.cc>
  $<${is-arm64}:v8/src/compiler/backend/arm64/instruction-selector-arm64.cc>
  $<${is-arm64}:v8/src/compiler/backend/arm64/unwinding-info-writer-arm64.cc>
  $<${is-arm64}:v8/src/deoptimizer/arm64/deoptimizer-arm64.cc>
  $<${is-arm64}:v8/src/diagnostics/arm64/disasm-arm64.cc>
  $<${is-arm64}:v8/src/diagnostics/arm64/eh-frame-arm64.cc>
  $<${is-arm64}:v8/src/execution/arm64/frame-constants-arm64.cc>
  $<${is-arm64}:v8/src/regexp/arm64/regexp-macro-assembler-arm64.cc>
  $<$<BOOL:${V8_ENABLE_I18N}>:$<TARGET_OBJECTS:v8-i18n-support>>
  ${api-sources}
  ${asmjs-sources}
  ${ast-sources}
  ${builtin-sources}
  ${baseline-sources}
  ${bigint-sources}
  ${codegen-sources}
  v8/src/common/assert-scope.cc
  v8/src/common/code-memory-access.cc
  v8/src/compiler-dispatcher/lazy-compile-dispatcher.cc
  v8/src/compiler-dispatcher/optimizing-compile-dispatcher.cc
  v8/src/date/date.cc
  v8/src/date/dateparser.cc
  ${debug-sources}
  ${deoptimizer-sources}
  ${diagnostic-sources}
  ${execution-sources}
  ${extensions-sources}
  v8/src/flags/flags.cc
  v8/src/handles/global-handles.cc
  v8/src/handles/handles.cc
  ${heap-sources}
  v8/src/ic/call-optimization.cc
  v8/src/ic/handler-configuration.cc
  v8/src/ic/ic-stats.cc
  v8/src/ic/ic.cc
  v8/src/ic/stub-cache.cc
  v8/src/init/bootstrapper.cc
  v8/src/init/icu_util.cc
  v8/src/init/isolate-allocator.cc
  v8/src/init/startup-data-util.cc
  v8/src/init/v8.cc
  ${handles-sources}
  ${interpreter-sources}
  ${json-sources}
  ${logging-sources}
  ${numbers-sources}
  ${object-sources}
  ${parsing-sources}
  ${profiler-sources}
  ${regexp-sources}
  v8/src/roots/roots.cc
  ${runtime-sources}
  v8/src/base/sanitizer/lsan-page-allocator.cc
  ${snapshot-sources}
  ${strings-sources}
  v8/src/tasks/cancelable-task.cc
  v8/src/tasks/operations-barrier.cc
  v8/src/tasks/task-utils.cc
  v8/src/third_party/siphash/halfsiphash.cc
  v8/src/tracing/trace-event.cc
  v8/src/tracing/traced-value.cc
  v8/src/tracing/tracing-category-observer.cc
  v8/src/trap-handler/handler-inside.cc
  v8/src/trap-handler/handler-outside.cc
  v8/src/trap-handler/handler-shared.cc
  v8/src/temporal/temporal-parser.cc
  ${utils-sources}
  ${wasm-sources}
  ${zone-sources}
  ${zlib-sources}
  $<TARGET_OBJECTS:v8-adler32>
)


target_compile_definitions(v8_base_without_compiler PRIVATE ${v8_defines} $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_base_without_compiler PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_base_without_compiler PRIVATE ${enable-fpic})
endif()

target_include_directories(v8_base_without_compiler
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/v8/include>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:v8-bytecodes-builtin-list,INTERFACE_INCLUDE_DIRECTORIES>>
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/zlib
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(
  v8_base_without_compiler
  PRIVATE
    v8_libbase
    v8_libsampler
    v8_torque_generated
  PUBLIC
    v8-bytecodes-builtin-list
)

#
# v8_compiler
#
file(GLOB_RECURSE compiler-sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS v8/src/compiler/*.cc
)
list(FILTER compiler-sources EXCLUDE REGEX ".*backend/.*/.*[.]cc$")

add_library(v8_compiler STATIC)
target_sources(v8_compiler PRIVATE ${compiler-sources})
target_compile_definitions(v8_compiler PRIVATE ${v8_defines} $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_compiler PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_compiler PRIVATE ${enable-fpic})
endif()

target_include_directories(v8_compiler
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:v8-bytecodes-builtin-list,INTERFACE_INCLUDE_DIRECTORIES>>
)

target_link_libraries(v8_compiler
  PRIVATE
    v8_libbase
    v8_base_without_compiler
    v8_torque_generated
  PUBLIC
    v8-bytecodes-builtin-list
)

#
# v8_initializers
#

add_library(
  v8_initializers STATIC
  $<${is-arm64}:v8/src/builtins/arm64/builtins-arm64.cc>
  $<${is-arm}:v8/src/builtins/arm/builtins-arm.cc>
  $<${is-ia32}:v8/src/builtins/ia32/builtins-ia32.cc>
  $<${is-mips64}:v8/src/builtins/mips64/builtins-mips64.cc>
  $<${is-mips}:v8/src/builtins/mips/builtins-mips.cc>
  $<${is-ppc}:v8/src/builtins/ppc/builtins-ppc.cc>
  $<${is-s390}:v8/src/builtins/s390/builtins-s390.cc>
  $<${is-x64}:v8/src/builtins/x64/builtins-x64.cc>
  $<$<BOOL:${V8_ENABLE_I18N}>:v8/src/builtins/builtins-intl-gen.cc>
  v8/src/builtins/builtins-array-gen.cc
  v8/src/builtins/builtins-async-function-gen.cc
  v8/src/builtins/builtins-async-gen.cc
  v8/src/builtins/builtins-async-generator-gen.cc
  v8/src/builtins/builtins-async-iterator-gen.cc
  v8/src/builtins/builtins-bigint-gen.cc
  v8/src/builtins/builtins-call-gen.cc
  v8/src/builtins/builtins-collections-gen.cc
  v8/src/builtins/builtins-constructor-gen.cc
  v8/src/builtins/builtins-conversion-gen.cc
  v8/src/builtins/builtins-date-gen.cc
  v8/src/builtins/builtins-generator-gen.cc
  v8/src/builtins/builtins-global-gen.cc
  v8/src/builtins/builtins-handler-gen.cc
  v8/src/builtins/builtins-ic-gen.cc
  v8/src/builtins/builtins-internal-gen.cc
  v8/src/builtins/builtins-interpreter-gen.cc
  v8/src/builtins/builtins-iterator-gen.cc
  v8/src/builtins/builtins-lazy-gen.cc
  v8/src/builtins/builtins-microtask-queue-gen.cc
  v8/src/builtins/builtins-number-gen.cc
  v8/src/builtins/builtins-object-gen.cc
  v8/src/builtins/builtins-promise-gen.cc
  v8/src/builtins/builtins-proxy-gen.cc
  v8/src/builtins/builtins-regexp-gen.cc
  v8/src/builtins/builtins-sharedarraybuffer-gen.cc
  v8/src/builtins/builtins-string-gen.cc
  v8/src/builtins/builtins-typed-array-gen.cc
  v8/src/builtins/builtins-wasm-gen.cc
  v8/src/builtins/growable-fixed-array-gen.cc
  v8/src/builtins/setup-builtins-internal.cc
  v8/src/codegen/code-stub-assembler.cc
  v8/src/heap/setup-heap-internal.cc
  v8/src/ic/accessor-assembler.cc
  v8/src/ic/binary-op-assembler.cc
  v8/src/ic/keyed-store-generic.cc
  v8/src/ic/unary-op-assembler.cc
  v8/src/interpreter/interpreter-assembler.cc
  v8/src/interpreter/interpreter-generator.cc
  v8/src/interpreter/interpreter-intrinsics-generator.cc
)

target_compile_definitions(v8_initializers PRIVATE ${v8_defines} $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_initializers PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_initializers PRIVATE ${enable-fpic})
endif()

target_include_directories(v8_initializers
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(v8_initializers PRIVATE v8_torque_generated v8-bytecodes-builtin-list)

#
# v8_snapshot
#

# Note: v8_use_external_startup_data not currently supported.
# Note: v8_use_multi_snapshots not currently supported.
add_library(
  v8_snapshot STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/embedded.S
  ${CMAKE_CURRENT_BINARY_DIR}/snapshot.cc
  v8/src/init/setup-isolate-deserialize.cc
)

target_compile_definitions(v8_snapshot PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_snapshot PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_snapshot PRIVATE ${enable-fpic})
endif()
target_include_directories(v8_snapshot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/v8)

target_link_libraries(v8_snapshot
  PRIVATE
    v8_torque_generated
    v8-bytecodes-builtin-list
)

# Note: allow passing in v8_random_seed
# we use mksnapshot here
# timeplus@yokofly cross compile binary mksnapshot
if (V8_CROSSCOMPILE)
add_custom_command(
  COMMAND
    ${CMAKE_CURRENT_SOURCE_DIR}/mksnapshot
    --embedded_src ${CMAKE_CURRENT_BINARY_DIR}/embedded.S
    --startup_src ${CMAKE_CURRENT_BINARY_DIR}/snapshot.cc
    $<${is-arm64}:--target_arch=arm64>
    $<${is-x64}:--target_arch=x64>
    $<$<PLATFORM_ID:Darwin>:--target_os=mac>
    $<$<PLATFORM_ID:Linux>:--target_os=linux>
    $<$<PLATFORM_ID:Windows>:--target_os=win>
    --turbo_instruction_scheduling
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/embedded.S
    ${CMAKE_CURRENT_BINARY_DIR}/snapshot.cc
)

else()
add_custom_command(
  COMMAND
    ${CMAKE_CURRENT_BINARY_DIR}/mksnapshot
    --embedded_src ${CMAKE_CURRENT_BINARY_DIR}/embedded.S
    --startup_src ${CMAKE_CURRENT_BINARY_DIR}/snapshot.cc
    $<${is-arm64}:--target_arch=arm64>
    $<${is-x64}:--target_arch=x64>
    $<$<PLATFORM_ID:Darwin>:--target_os=mac>
    $<$<PLATFORM_ID:Linux>:--target_os=linux>
    $<$<PLATFORM_ID:Windows>:--target_os=win>
    --turbo_instruction_scheduling
  DEPENDS
    mksnapshot
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/embedded.S
    ${CMAKE_CURRENT_BINARY_DIR}/snapshot.cc
)
endif()

#
# v8_inspector
#

# Note: this is the inspector, broken out for ease of maintenance.
# The GN files are conceptually structured the same but they don't
# build it as a separate library. We do because it's easier that way.
set(inspector_files
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/include/inspector/Debugger.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/include/inspector/Runtime.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/include/inspector/Schema.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Console.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Console.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Debugger.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Debugger.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Forward.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/HeapProfiler.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/HeapProfiler.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Profiler.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Profiler.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Protocol.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Protocol.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Runtime.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Runtime.h
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Schema.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector/protocol/Schema.h
)

foreach(filename IN LISTS inspector_files)
  get_filename_component(directory ${filename} DIRECTORY)
  list(APPEND inspector_dirs ${directory})
endforeach()
list(REMOVE_DUPLICATES inspector_dirs)

file(GLOB inspector-sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS v8/src/inspector/*.cc
)

file(GLOB inspector-protocol-sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS v8/third_party/inspector_protocol/crdtp/*.cc
)

list(FILTER inspector-protocol-sources EXCLUDE REGEX "_test[.]cc$")
list(FILTER inspector-protocol-sources EXCLUDE REGEX "crdtp/test_.*[.]cc$")

add_library(v8_inspector STATIC
  ${inspector-protocol-sources}
  ${inspector-sources}
  ${inspector_files}
)

target_compile_features(v8_inspector PUBLIC cxx_std_17)
target_compile_options(v8_inspector PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_inspector PRIVATE ${enable-fpic})
endif()
target_compile_definitions(v8_inspector PUBLIC $<${is-msvc}:_HAS_EXCEPTIONS=0>)

target_include_directories(v8_inspector
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/inspector
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/googletest/src/googlemock/include
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/googletest/src/googletest/include
)

target_link_libraries(v8_inspector PRIVATE v8_torque_generated)

file(GLOB inspector-templates
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS
    v8/third_party/inspector_protocol/lib/*.template
    v8/third_party/inspector_protocol/templates/*.template
)

add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${inspector_dirs}
  OUTPUT ${inspector_dirs}
)

add_custom_command(
  COMMAND
    Python::Interpreter
    v8/third_party/inspector_protocol/code_generator.py
    --config v8/src/inspector/inspector_protocol_config.json
    --config_value protocol.path=v8/include/js_protocol.pdl
    --jinja_dir v8/third_party
    --output_base ${CMAKE_CURRENT_BINARY_DIR}/inspector/src/inspector
    --inspector_protocol_dir third_party/inspector_protocol
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    v8/third_party/inspector_protocol/code_generator.py
    v8/src/inspector/inspector_protocol_config.json
    ${inspector-templates}
    ${inspector_dirs}
  OUTPUT
    ${inspector_files}
)

#
# v8_libplatform
#

add_library(v8_libplatform STATIC)
target_sources(v8_libplatform
  PRIVATE
    $<${is-win}:v8/src/libplatform/tracing/recorder-win.cc>
    v8/src/libplatform/default-foreground-task-runner.cc
    v8/src/libplatform/default-job.cc
    v8/src/libplatform/default-platform.cc
    v8/src/libplatform/default-thread-isolated-allocator.cc
    v8/src/libplatform/default-worker-threads-task-runner.cc
    v8/src/libplatform/delayed-task-queue.cc
    v8/src/libplatform/task-queue.cc
    v8/src/libplatform/tracing/trace-buffer.cc
    v8/src/libplatform/tracing/trace-config.cc
    v8/src/libplatform/tracing/trace-object.cc
    v8/src/libplatform/tracing/trace-writer.cc
    v8/src/libplatform/tracing/tracing-controller.cc
    v8/src/libplatform/worker-thread.cc
)

target_compile_definitions(v8_libplatform PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_libplatform PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_libplatform PRIVATE ${enable-fpic})
endif()

target_include_directories(v8_libplatform
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)

target_link_libraries(v8_libplatform PRIVATE v8_libbase)

#
# v8_libsampler
#

add_library(v8_libsampler STATIC v8/src/libsampler/sampler.cc)

target_include_directories(v8_libsampler
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
)

target_compile_definitions(v8_libsampler PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_libsampler PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_libsampler PRIVATE ${enable-fpic})
endif()
target_include_directories(v8_libsampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/v8)
target_link_libraries(v8_libsampler PRIVATE v8_libbase)

#
# v8_libbase
#

add_library(v8_libbase STATIC
  v8/src/base/bits.cc
  v8/src/base/bounded-page-allocator.cc
  v8/src/base/cpu.cc
  v8/src/base/debug/stack_trace.cc
  v8/src/base/division-by-constant.cc
  v8/src/base/file-utils.cc
  v8/src/base/ieee754.cc
  v8/src/base/logging.cc
  v8/src/base/once.cc
  v8/src/base/page-allocator.cc
  v8/src/base/platform/condition-variable.cc
  v8/src/base/platform/memory-protection-key.cc
  v8/src/base/platform/mutex.cc
  v8/src/base/platform/platform.cc
  v8/src/base/platform/semaphore.cc
  v8/src/base/platform/time.cc
  v8/src/base/region-allocator.cc
  v8/src/base/sys-info.cc
  v8/src/base/strings.cc
  v8/src/base/utils/random-number-generator.cc
  v8/src/base/vlq-base64.cc
  v8/src/base/virtual-address-space.cc
  $<$<NOT:$<OR:${is-win},${is-aix}>>:v8/src/base/platform/platform-posix-time.cc>
  $<${is-win}:v8/src/base/platform/platform-win32.cc>
  $<${is-win}:v8/src/base/debug/stack_trace_win.cc>
  $<${is-aix}:v8/src/base/debug/stack_trace_posix.cc>
  $<${is-aix}:v8/src/base/platform/platform-aix.cc>
  $<${is-darwin}:v8/src/base/debug/stack_trace_posix.cc>
  $<${is-darwin}:v8/src/base/platform/platform-darwin.cc>
  $<${is-linux}:v8/src/base/debug/stack_trace_posix.cc>
  $<${is-linux}:v8/src/base/platform/platform-linux.cc>
  $<${is-posix}:v8/src/base/platform/platform-posix.cc>
)

set_property(SOURCE v8/src/base/utils/random-number-generator.cc
  APPEND PROPERTY COMPILE_DEFINITIONS _CRT_RAND_S
)

target_compile_definitions(v8_libbase PRIVATE $<${is-win}:UNICODE> $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_libbase PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_libbase PRIVATE ${enable-fpic})
endif()
target_include_directories(v8_libbase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/v8)
target_link_libraries(v8_libbase
  PRIVATE
    Threads::Threads
    $<${is-linux}:${CMAKE_DL_LIBS}>
    $<${is-linux}:rt>
    $<${is-win}:winmm>
    $<${is-win}:dbghelp>
)

#
# bytecode_builtins_list_generator
#
# we use bytecode_builtins_list_generator here
# timeplus@yokofly cross compile binary bytecode_builtins_list_generator
if (V8_CROSSCOMPILE)
  message(STATUS "V8 Cross compile build for bytecode_builtins_list_generator")
  
  # First verify the pre-built tool exists
  set(PREBUILT_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/bytecode_builtins_list_generator")
  if(NOT EXISTS ${PREBUILT_GENERATOR})
    message(FATAL_ERROR "Cross compilation requires pre-built bytecode_builtins_list_generator at ${PREBUILT_GENERATOR}")
  endif()
  
  function(v8_generate_builtins_list target-dir)
    set(directory ${target-dir}/builtins-generated)
    set(output ${directory}/bytecodes-builtins-list.h)
    
    add_custom_command(
      COMMAND ${CMAKE_COMMAND} -E make_directory ${directory}
      OUTPUT ${directory}
      COMMENT "Generating ${directory}"
      VERBATIM)

    add_custom_command(
      COMMAND ${PREBUILT_GENERATOR} ${output}
      DEPENDS ${directory}  # Removed bytecode_builtins_list_generator from DEPENDS
      OUTPUT ${output}
      COMMENT "Generating ${output}"
      VERBATIM)

    add_library(v8-bytecodes-builtin-list INTERFACE)
    target_include_directories(v8-bytecodes-builtin-list INTERFACE ${target-dir})
    target_sources(v8-bytecodes-builtin-list INTERFACE ${output})
  endfunction()

else()
  message(STATUS "V8 Native build for bytecode_builtins_list_generator !!!")
  function(v8_generate_builtins_list target-dir)
    set(directory ${target-dir}/builtins-generated)
    set(output ${directory}/bytecodes-builtins-list.h)
    add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${directory}
    OUTPUT ${directory}
    COMMENT "Generating ${directory}"
    VERBATIM)
  add_custom_command(
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bytecode_builtins_list_generator ${output}
    DEPENDS ${directory} bytecode_builtins_list_generator
    OUTPUT ${output}
    COMMENT "Generating ${output}"
    VERBATIM)
  add_library(v8-bytecodes-builtin-list INTERFACE)
  target_include_directories(v8-bytecodes-builtin-list INTERFACE ${target-dir})
    target_sources(v8-bytecodes-builtin-list INTERFACE ${output})
  endfunction()
endif()

v8_generate_builtins_list(${CMAKE_CURRENT_BINARY_DIR}/generated)


# FIXME(yokofly)
# Cross-compilation requires running intermediate binaries on the host.
# So we skip native compilation and use a pre-built binary(bytecode_builtins_list_generator) instead.
if (NOT V8_CROSSCOMPILE)
message(STATUS "V8 Native build for bytecode_builtins_list_generator !!!")
target_include_directories(v8-bytecodes-builtin-list
  INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

add_executable(
  bytecode_builtins_list_generator
  v8/src/builtins/generate-bytecodes-builtins-list.cc
  v8/src/interpreter/bytecode-operands.cc
  v8/src/interpreter/bytecodes.cc
)

target_include_directories(bytecode_builtins_list_generator
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)

target_link_libraries(bytecode_builtins_list_generator v8_libbase)
endif()

#
# v8_torque_generated
#

# Note: torque does not like absolute paths.
file(GLOB_RECURSE torque-builtins
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/v8
  CONFIGURE_DEPENDS v8/src/builtins/*.tq
)

file(GLOB_RECURSE torque-objects
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/v8
  CONFIGURE_DEPENDS v8/src/objects/*.tq
)

list(APPEND torque_files ${torque-builtins})
list(APPEND torque_files ${torque-objects})

list(APPEND torque_files
  src/debug/debug-wasm-objects.tq
  src/ic/handler-configuration.tq
  src/wasm/wasm-objects.tq
  test/torque/test-torque.tq
  third_party/v8/builtins/array-sort.tq
)

if(NOT V8_ENABLE_I18N)
  list(REMOVE_ITEM torque_files
    src/objects/intl-objects.tq
    src/objects/js-break-iterator.tq
    src/objects/js-collator.tq
    src/objects/js-date-time-format.tq
    src/objects/js-duration-format.tq
    src/objects/js-display-names.tq
    src/objects/js-list-format.tq
    src/objects/js-locale.tq
    src/objects/js-number-format.tq
    src/objects/js-plural-rules.tq
    src/objects/js-relative-time-format.tq
    src/objects/js-segment-iterator.tq
    src/objects/js-segmenter.tq
    src/objects/js-segments.tq
  )
endif()

list(TRANSFORM torque_files PREPEND v8/ OUTPUT_VARIABLE torque_files_abs)

# Note: target v8_compiler depends on ${torque_outputs}. Maybe this should be
# a source-only dependency.
set(torque_outputs
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/class-debug-readers-tq.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/class-debug-readers.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/class-definitions-tq.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/class-definitions.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/class-verifiers-tq.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/class-verifiers.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/exported-macros-assembler-tq.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/exported-macros-assembler.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/factory-tq.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/factory-tq.inc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/factory.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/factory.inc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/interface-descriptors-tq.inc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/interface-descriptors.inc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/objects-body-descriptors-inl.inc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/objects-body-descriptors-tq-inl.inc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/objects-printer-tq.cc
  ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/objects-printer.cc
)

list(TRANSFORM torque_files
  REPLACE "[.]tq$" "-tq-csa.cc"
  OUTPUT_VARIABLE torque-outputs
)

list(TRANSFORM torque-outputs
  REPLACE "[.]cc$" ".h"
  OUTPUT_VARIABLE torque-headers
)

list(APPEND torque-outputs ${torque-headers})
list(TRANSFORM torque-outputs PREPEND "${CMAKE_CURRENT_BINARY_DIR}/torque-generated/")

foreach(filename IN LISTS torque_files)
  get_filename_component(directory ${filename} DIRECTORY)
  list(APPEND torque_dirs ${CMAKE_CURRENT_BINARY_DIR}/torque-generated/${directory})
endforeach()

list(REMOVE_DUPLICATES torque_dirs)

add_library(
  v8_torque_generated STATIC
  ${torque-outputs}
  ${torque_outputs}
)

target_compile_definitions(v8_torque_generated PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_torque_generated PRIVATE ${disable-exceptions})
if(enable-fPIC)
  target_compile_options(v8_torque_generated PRIVATE ${enable-fpic})
endif()
target_link_libraries(v8_torque_generated PRIVATE v8-bytecodes-builtin-list)

target_include_directories(v8_torque_generated
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)

# we use torque here
if (NOT V8_CROSSCOMPILE)
add_custom_command(
  COMMAND
    ${CMAKE_CURRENT_BINARY_DIR}/torque
    -o ${CMAKE_CURRENT_BINARY_DIR}/torque-generated
    -v8-root ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${torque_files}
  COMMAND
    ${CMAKE_COMMAND} -E touch ${torque-outputs} ${torque_outputs}
  DEPENDS
    torque
    ${torque_dirs}
    ${torque_files_abs}
  OUTPUT
    ${torque-outputs}
    ${torque_outputs}
)
else()
set(PREBUILT_TORQUE "${CMAKE_CURRENT_SOURCE_DIR}/torque")
  if(NOT EXISTS ${PREBUILT_TORQUE})
    message(FATAL_ERROR "Cross compilation requires pre-built torque at ${PREBUILT_TORQUE}")
  endif()
  
  add_custom_command(
    COMMAND
      ${PREBUILT_TORQUE}
      -o ${CMAKE_CURRENT_BINARY_DIR}/torque-generated
      -v8-root ${CMAKE_CURRENT_SOURCE_DIR}/v8
      ${torque_files}
    COMMAND
      ${CMAKE_COMMAND} -E touch ${torque-outputs} ${torque_outputs}
    DEPENDS
      ${torque_dirs}
      ${torque_files_abs}
    OUTPUT
      ${torque-outputs}
      ${torque_outputs}
    COMMENT "Generating torque outputs using pre-built torque"
    VERBATIM
  )

endif()

add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${torque_dirs}
  OUTPUT
    ${torque_dirs}
)

#
# torque
#
# FIXME(yokofly)
# Cross-compilation requires running intermediate binaries on the host.
# So we skip native compilation and use a pre-built binary(torque) instead.
if (NOT V8_CROSSCOMPILE)
file(GLOB torque-program-sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS v8/src/torque/*.cc
)

add_executable(torque)
target_sources(torque PRIVATE ${torque-program-sources})
target_compile_options(torque PRIVATE ${enable-exceptions})
target_include_directories(torque PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/v8)
target_link_libraries(torque PRIVATE v8_libbase)
endif()
#
# mksnapshot
#

# FIXME(yokofly)
# Cross-compilation requires running intermediate binaries on the host.
# So we skip native compilation and use a pre-built binary(mksnapshot) instead.

if (NOT V8_CROSSCOMPILE)
add_executable(mksnapshot
  v8/src/init/setup-isolate-full.cc
  v8/src/snapshot/embedded/embedded-empty.cc
  v8/src/snapshot/embedded/embedded-file-writer.cc
  v8/src/snapshot/embedded/platform-embedded-file-writer-aix.cc
  v8/src/snapshot/embedded/platform-embedded-file-writer-base.cc
  v8/src/snapshot/embedded/platform-embedded-file-writer-generic.cc
  v8/src/snapshot/embedded/platform-embedded-file-writer-mac.cc
  v8/src/snapshot/embedded/platform-embedded-file-writer-win.cc
  v8/src/snapshot/mksnapshot.cc
  v8/src/snapshot/snapshot-empty.cc
)

target_compile_definitions(mksnapshot PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(mksnapshot PRIVATE ${disable-exceptions})
target_include_directories(mksnapshot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/v8)
target_link_libraries(mksnapshot
  PRIVATE
    v8_libbase
    v8_libplatform
    v8_base_without_compiler
    v8_compiler
    v8_initializers
    v8-bytecodes-builtin-list
    v8_torque_generated
)
endif()

add_library(v8-adler32 OBJECT v8/third_party/zlib/adler32.c)
target_compile_options(v8-adler32 PRIVATE ${disable-exceptions})
target_include_directories(v8-adler32
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/zlib>
)

add_library(ch_contrib::v8_base_without_compiler ALIAS v8_base_without_compiler)
add_library(ch_contrib::v8_compiler              ALIAS v8_compiler             )
add_library(ch_contrib::v8_snapshot              ALIAS v8_snapshot             )
add_library(ch_contrib::v8_inspector             ALIAS v8_inspector            )
add_library(ch_contrib::v8_libplatform           ALIAS v8_libplatform          )
add_library(ch_contrib::v8_libbase               ALIAS v8_libbase              )
add_library(ch_contrib::v8_libsampler            ALIAS v8_libsampler           )
add_library(ch_contrib::v8_initializers          ALIAS v8_initializers         )
add_library(ch_contrib::v8_torque_generated      ALIAS v8_torque_generated     )
