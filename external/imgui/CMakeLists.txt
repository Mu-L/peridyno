set(LIB_NAME imgui)
file(GLOB_RECURSE IMGUI_SRC
     LIST_DIRECTORIES false
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
	 "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(imgui SHARED ${IMGUI_SRC})
set_target_properties(imgui PROPERTIES
    OUTPUT_NAME "imgui-1.90.8")

set_target_properties(imgui PROPERTIES FOLDER "ThirdParty")

if(WIN32)
    set_target_properties(imgui PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(imgui PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(imgui PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()

target_include_directories(imgui PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(imgui glad)

install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/external/imgui)