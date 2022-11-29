cmake_minimum_required(VERSION 3.10)

option(V8CMAKE_FIND_PYTHON "Enable finding python" ON)

add_definitions("-DV8_ENABLE_WEBASSEMBLY -DV8_ADVANCED_BIGINT_ALGORITHMS")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (V8CMAKE_FIND_PYTHON)
  find_package(Python 3 REQUIRED QUIET COMPONENTS Interpreter)
endif()
find_package(Threads QUIET)

include(CMakeDependentOption)
include(GNUInstallDirs)

include(CheckPythonModuleExists)
include(GenerateBuiltinsList)

check_python_module_exists(PYTHON_HAVE_MARKUPSAFE markupsafe)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
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
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
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
  $<$<CXX_COMPILER_ID:Clang>:-fno-exceptions -Wno-undef -Wno-unused-parameter -Wno-covered-switch-default -Wno-poison-system-directories -Wno-gnu-statement-expression -Wno-comma -Wno-format-nonliteral -Wno-format-pedantic -Wno-missing-noreturn -Wno-conditional-uninitialized -Wno-extra-semi-stmt -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-undef -Wno-shadow-field-in-constructor -Wno-shadow-field-in-constructor -Wno-documentation -Wno-reserved-identifier -Wno-shadow-field-in-constructor -Wno-newline-eof -Wno-undefined-reinterpret-cast >
  $<$<CXX_COMPILER_ID:GNU>:-fno-exceptions>
)

set(enable-exceptions
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
  $<$<CXX_COMPILER_ID:AppleClang>:-fexceptions>
  $<$<CXX_COMPILER_ID:Clang>:-fexceptions>
  $<$<CXX_COMPILER_ID:GNU>:-fexceptions>
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/generated)
add_library(v8-bytecodes-builtin-list STATIC ${CMAKE_CURRENT_SOURCE_DIR}/generated/builtins-generated/bytecodes-builtins-list.h)
# more granular library support
add_library(v8-i18n-support OBJECT)
set_property(TARGET v8-i18n-support PROPERTY EXCLUDE_FROM_ALL ON)

list(APPEND i18n-sources
  v8/src/builtins/builtins-intl.cc
  v8/src/objects/intl-objects.cc
  v8/src/objects/js-break-iterator.cc
  v8/src/objects/js-collator.cc
  v8/src/objects/js-date-time-format.cc
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
  PUBLIC
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
list(REMOVE_ITEM diagnostic-sources v8/src/diagnostics/system-jit-win.cc)
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
  $<$<AND:${is-x64},${is-win}>:v8/src/diagnostics/system-jit-win.cc>
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
  v8/src/web-snapshot/web-snapshot.cc
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
target_compile_options(v8_base_without_compiler 
  PRIVATE ${disable-exceptions} 
  PUBLIC -Wno-error -Wno-implicit-fallthrough -Wno-unused-function -Wno-sign-compare -Wno-deprecated-declarations -Wno-deprecated-pragma -Wno-redundant-parens -Wno-deprecated-anon-enum-enum-conversion -Wno-zero-as-null-pointer-constant -Wno-suggest-override -Wno-dtor-name -Wno-suggest-destructor-override -Wno-inconsistent-missing-destructor-override -Wno-frame-larger-than -Wno-unused-macros -Wno-deprecated-volatile -Wno-cast-function-type -Wno-deprecated-copy-with-dtor -Wno-deprecated-this-capture -Wno-unused-member-function -Wno-shadow -Wno-gnu-anonymous-struct -Wno-shadow-field -Wno-cast-qual -Wno-used-but-marked-unused -Wno-gnu-zero-variadic-macro-arguments -Wno-ambiguous-reversed-operator -Wno-deprecated-copy-with-user-provided-dtor -Wno-shadow-uncaptured-local -Wno-ambiguous-reversed-operator -Wno-duplicate-enum -Wno-old-style-cast
)

target_include_directories(v8_base_without_compiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/v8/include>
   # $<BUILD_INTERFACE:$<TARGET_PROPERTY:v8-bytecodes-builtin-list,INTERFACE_INCLUDE_DIRECTORIES>>
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
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
target_compile_options(v8_compiler
  PRIVATE ${disable-exceptions} 
  PUBLIC -Wno-error -Wno-shadow -Wno-shadow -Wno-tautological-type-limit-compare -Wno-deprecated-this-capture -Wno-unused-member-function -Wno-unused-macros -Wno-deprecated-copy-with-dtor -Wno-gnu-anonymous-struct -Wno-suggest-override -Wno-deprecated-copy-with-user-provided-dtor -Wno-deprecated-volatile -Wno-shadow-uncaptured-local -Wno-shadow-field -Wno-ambiguous-reversed-operator -Wno-deprecated-volatile -Wno-duplicate-enum -Wno-old-style-cast -Wno-used-but-marked-unused -Wno-cast-qual -Wno-gnu-zero-variadic-macro-arguments
)

target_include_directories(v8_compiler
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
   # $<BUILD_INTERFACE:$<TARGET_PROPERTY:v8-bytecodes-builtin-list,INTERFACE_INCLUDE_DIRECTORIES>>
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
target_compile_options(v8_initializers 
  PRIVATE ${disable-exceptions} 
  PUBLIC -Wno-error -Wno-shadow -Wno-duplicate-enum -Wno-frame-larger-than -Wno-gnu-anonymous-struct -Wno-deprecated-this-capture -Wno-ambiguous-reversed-operator -Wno-shadow-field -Wno-shadow-uncaptured-local -Wno-ambiguous-reversed-operator -Wno-old-style-cast -Wno-deprecated-copy-with-dtor -Wno-unused-member-function -Wno-unused-macros -Wno-gnu-zero-variadic-macro-arguments -Wno-cast-qual  -Wno-used-but-marked-unused -Wno-duplicate-enum
)

target_include_directories(v8_initializers
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
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
target_compile_options(v8_snapshot 
  PRIVATE ${disable-exceptions} 
  PUBLIC -Wno-error -Wno-old-style-cast
)
target_include_directories(v8_snapshot 
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/v8
  ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
)

target_link_libraries(v8_snapshot
  PRIVATE
    v8_torque_generated
    v8-bytecodes-builtin-list
)

# Note: allow passing in v8_random_seed
add_custom_command(
  COMMAND
    ./mksnapshot
    --embedded_src ${CMAKE_CURRENT_BINARY_DIR}/embedded.S
    --startup_src ${CMAKE_CURRENT_BINARY_DIR}/snapshot.cc
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
target_compile_options(v8_inspector 
  PRIVATE ${disable-exceptions} 
  PUBLIC -Wno-error -Wno-cast-function-type -Wno-shadow -Wno-suggest-destructor-override -Wno-shadow-uncaptured-local -Wno-deprecated-copy-with-user-provided-dtor -Wno-shadow-field -Wno-gnu-zero-variadic-macro-arguments -Wno-old-style-cast -Wno-inconsistent-missing-destructor-override -Wno-unused-member-function -Wno-unused-macros -Wno-used-but-marked-unused)
target_compile_definitions(v8_inspector PUBLIC $<${is-msvc}:_HAS_EXCEPTIONS=0>)

target_include_directories(v8_inspector
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/inspector
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/googletest/src/googlemock/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/googletest/src/googletest/include
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
target_compile_options(v8_libplatform PRIVATE ${disable-exceptions} PUBLIC -Wno-error -Wno-unused-parameter -Wno-non-virtual-dtor)
target_include_directories(v8_libplatform
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)

target_link_libraries(v8_libplatform PRIVATE v8_libbase)

#
# v8_libsampler
#

add_library(v8_libsampler STATIC v8/src/libsampler/sampler.cc)

target_include_directories(v8_libsampler
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
)

target_compile_definitions(v8_libsampler PRIVATE $<${is-msvc}:_HAS_EXCEPTIONS=0>)
target_compile_options(v8_libsampler PRIVATE ${disable-exceptions})
target_include_directories(v8_libsampler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/v8)
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
  v8/src/base/functional.cc
  v8/src/base/ieee754.cc
  v8/src/base/logging.cc
  v8/src/base/once.cc
  v8/src/base/page-allocator.cc
  v8/src/base/platform/condition-variable.cc
  v8/src/base/platform/mutex.cc
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
  $<${is-darwin}:v8/src/base/platform/platform-macos.cc>
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
target_include_directories(v8_libbase 
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
)
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

add_executable(
  bytecode_builtins_list_generator
  v8/src/builtins/generate-bytecodes-builtins-list.cc
  v8/src/interpreter/bytecode-operands.cc
  v8/src/interpreter/bytecodes.cc
)

target_include_directories(bytecode_builtins_list_generator
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)

target_link_libraries(bytecode_builtins_list_generator v8_libbase)

target_compile_options(bytecode_builtins_list_generator PUBLIC -Wno-error -Wno-unused-parameter -Wno-undef -Wno-shadow-field-in-constructor -Wno-duplicate-enum)
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
target_compile_options(v8_torque_generated PRIVATE ${disable-exceptions} PUBLIC -Wno-error -Wno-deprecated-copy-with-dtor -Wno-ambiguous-reversed-operator -Wno-deprecated-copy-with-user-provided-dtor -Wno-shadow-uncaptured-local -Wno-shadow-field -Wno-ambiguous-reversed-operator -Wno-duplicate-enum -Wno-old-style-cast -Wno-used-but-marked-unused -Wno-cast-qual -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments)
target_link_libraries(v8_torque_generated PRIVATE v8-bytecodes-builtin-list)

target_include_directories(v8_torque_generated
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
    ${CMAKE_CURRENT_SOURCE_DIR}/v8
)

add_custom_command(
  COMMAND
    ./torque
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

add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${torque_dirs}
  OUTPUT
    ${torque_dirs}
)

#
# torque
#
file(GLOB torque-program-sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_DEPENDS v8/src/torque/*.cc
)

add_executable(torque)
target_sources(torque PRIVATE ${torque-program-sources})
target_compile_options(torque PRIVATE ${enable-exceptions} PUBLIC -Wno-error -Wno-cast-qual -Wno-used-but-marked-unused -Wno-duplicate-enum -Wno-gnu-zero-variadic-macro-arguments -Wno-old-style-cast -Wno-gnu-anonymous-struct -Wno-cast-qual -Wno-used-but-marked-unused -Wno-duplicate-enum -Wno-deprecated-this-capture -Wno-extra-semi-stmt -Wno-missing-noreturn -Wno-used-but-marked-unused -Wno-conditional-uninitialized -Wno-deprecated-copy-with-dtor -Wno-covered-switch-default -Wno-shadow-field -Wno-shadow-field-in-constructor -Wno-unused-parameter -Wno-undef -Wno-newline-eof -Wno-non-virtual-dtor -Wno-documentation)
target_include_directories(torque PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/v8 ${CMAKE_CURRENT_SOURCE_DIR}/v8/include)
target_link_libraries(torque PRIVATE v8_libbase)

#
# mksnapshot
#

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
target_compile_options(mksnapshot PRIVATE ${disable-exceptions} PUBLIC -Wno-error -Wno-old-style-cast -Wno-frame-larger-than -Wno-unused-macros -Wno-deprecated-copy-with-user-provided-dtor -Wno-shadow-field  -Wno-gnu-zero-variadic-macro-arguments -Wno-shadow-uncaptured-local -Wno-old-style-cast -Wno-duplicate-enum -Wno-ambiguous-reversed-operator -Wno-used-but-marked-unused -Wno-cast-qual -Wno-gnu-anonymous-struct)
target_include_directories(mksnapshot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/v8 ${CMAKE_CURRENT_SOURCE_DIR}/v8/include)
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

add_library(v8-adler32 OBJECT v8/third_party/zlib/adler32.c)
target_include_directories(v8-adler32
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/v8/third_party/zlib>
    ${CMAKE_CURRENT_SOURCE_DIR}/v8 
    ${CMAKE_CURRENT_SOURCE_DIR}/v8/include
)

target_compile_options(v8-adler32 PRIVATE -Wno-extra-semi-stmt)

add_library(ch_contrib::v8_base_without_compiler ALIAS v8_base_without_compiler)
add_library(ch_contrib::v8_compiler              ALIAS v8_compiler             )
add_library(ch_contrib::v8_snapshot              ALIAS v8_snapshot             )
add_library(ch_contrib::v8_inspector             ALIAS v8_inspector            )
add_library(ch_contrib::v8_libplatform           ALIAS v8_libplatform          )
add_library(ch_contrib::v8_libbase               ALIAS v8_libbase              )
add_library(ch_contrib::v8_libsampler            ALIAS v8_libsampler           )
add_library(ch_contrib::v8_initializers          ALIAS v8_initializers         )
add_library(ch_contrib::v8_torque_generated      ALIAS v8_torque_generated     )