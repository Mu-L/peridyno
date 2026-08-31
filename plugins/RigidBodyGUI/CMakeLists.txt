option(PERIDYNO_PLUGIN_RIGIDBODYGUI "Enable binding the RigidBodyGUI library" OFF)

if(PERIDYNO_PLUGIN_RIGIDBODYGUI)
    set(LIB_DEPENDENCY Core Framework Topology GLRenderEngine IO RigidBody QtGUI FBXLoader Modeling)
    add_plugin(RigidBodyGUI LIB_DEPENDENCY)
endif()
