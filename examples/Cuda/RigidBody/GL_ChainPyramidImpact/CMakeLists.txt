if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY
        RigidBody
        ObjIO
        UbiGUI)
    add_example(GL_ChainPyramidImpact RigidBody LIB_DEPENDENCY)
endif()
