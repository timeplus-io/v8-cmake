function(v8_generate_builtins_list target-dir)
  # Expects V8_BYTECODE_BUILTINS_LIST_GENERATOR to point to the host tool.
  # For native builds, set it to ${CMAKE_CURRENT_BINARY_DIR}/bytecode_builtins_list_generator
  # For cross builds, set it to ${CMAKE_CURRENT_SOURCE_DIR}/bytecode_builtins_list_generator or a custom path.
  if (NOT V8_BYTECODE_BUILTINS_LIST_GENERATOR)
    message(FATAL_ERROR "V8_BYTECODE_BUILTINS_LIST_GENERATOR is not set")
  endif()
  set(directory ${target-dir}/builtins-generated)
  set(output ${directory}/bytecodes-builtins-list.h)
  add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${directory}
    OUTPUT ${directory}
    COMMENT "Generating ${directory}"
    VERBATIM)
  set(_v8_builtins_depends ${directory})
  if (TARGET bytecode_builtins_list_generator)
    list(APPEND _v8_builtins_depends bytecode_builtins_list_generator)
  endif()
  add_custom_command(
    COMMAND ${V8_BYTECODE_BUILTINS_LIST_GENERATOR} ${output}
    DEPENDS ${_v8_builtins_depends}
    OUTPUT ${output}
    COMMENT "Generating ${output}"
    VERBATIM)
  add_library(v8-bytecodes-builtin-list INTERFACE)
  target_include_directories(v8-bytecodes-builtin-list INTERFACE ${target-dir})
  target_sources(v8-bytecodes-builtin-list INTERFACE ${output})
endfunction()
