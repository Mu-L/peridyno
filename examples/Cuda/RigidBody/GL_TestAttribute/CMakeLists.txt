if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        GlfwGUI)
    add_example(GL_TestAttribute RigidBody LIB_DEPENDENCY)
endif()