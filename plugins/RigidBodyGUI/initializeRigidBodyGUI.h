#pragma once

#include "Plugin/PluginEntry.h"

namespace dyno
{
	/**
	 * RigidBodyGUI plugin initializer.
	 * Automatically registers "Open" and "Create Rigid Body" actions to QContentBrowser.
	 */
	class RigidBodyGUIInitializer : public PluginEntry
	{
	public:
		static PluginEntry* instance();
		static std::vector<std::shared_ptr<class Node>> assetNodes;

	protected:
		void initializeActions() override;

	private:
		RigidBodyGUIInitializer() {}

		static std::atomic<RigidBodyGUIInitializer*> gInstance;
		static std::mutex gMutex;

	};
}

namespace RigidBodyGUI
{
	// For static linking
	dyno::PluginEntry* initStaticPlugin();

	// For dynamic loading (plugin entry point)
	PERIDYNO_API dyno::PluginEntry* initDynoPlugin();
}