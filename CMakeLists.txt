cmake_minimum_required(VERSION 3.20)

# FIXME(github@yokofly)
# Cross-compilation requires running intermediate binaries on the host.
# So we skip native compilation and use a pre-built binary(bytecode_builtins_list_generator/torque/mksnapshot) instead.
# we can replace with this style in future https://github.com/ClickHouse/ClickHouse/pull/38936/commits/b7665524817281f261461a24893a67e3a2806216

if (NOT CMAKE_CROSSCOMPILING)
  include(CMakeLists_default.txt)
else()
  include(CMakeLists_crosscompile.txt)
endif()
