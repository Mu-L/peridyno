set(TEST_PROJECT Test_Volume)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_executable(${TEST_PROJECT} ${TEST_SOURCES})
target_link_libraries(${TEST_PROJECT} PUBLIC 
    gtest 
    Core 
    Framework 
    Volume)

add_test(NAME ${TEST_PROJECT} COMMAND ${TEST_PROJECT})

set_target_properties(${TEST_PROJECT} PROPERTIES FOLDER "Tests")