if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        QtGUI
        ObjIO
        GlfwGUI
        UbiGUI
        FBXLoader
    )
    add_example(Qt_Arm RigidBody LIB_DEPENDENCY)
endif()