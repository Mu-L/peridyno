#include "initializeRigidBodyGUI.h"
#include "PConsoleWidget.h"
#include "RigidBody/MultiBodyTuple.h"
#include "NodeFactory.h"
#include "ViewerItem/PTextureMeshViewerWidget.h"
#include "Topology/TextureMeshInterface.h"
#include "Platform.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QToolButton>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QRegularExpression>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "Node.h"
#include "QtGUI/PPropertyWidget.h"
#include "RigidBody/ConfigurableBody.h"
#include "FBXLoader/FBXLoader.h"
#include "GltfLoader.h"
#include "TextureMeshLoader.h"
#include "Topology/TextureMeshInterface.h"
#include "QtGUI/ViewerItem/GLMeshRenderWidget.h"
#include "RigidBody/ConfigurableBody.h"
#include <QCheckBox>
#include <QSplitter>
#include "LockerButton.h"
#include "Platform.h"

namespace dyno
{
	// Forward declare callback functions (defined below)
	static void openMeshCallback(const QString& filePath, QWidget* parent);
	static void createRigidBodyCallback(const QString& filePath, QWidget* parent);
	static void editPDMCallback(const QString& filePath, QWidget* parent);

	std::atomic<RigidBodyGUIInitializer*> RigidBodyGUIInitializer::gInstance;
	std::mutex RigidBodyGUIInitializer::gMutex;
	std::vector<std::shared_ptr<Node>> RigidBodyGUIInitializer::assetNodes;

	PluginEntry* RigidBodyGUIInitializer::instance()
	{
		RigidBodyGUIInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new RigidBodyGUIInitializer();
				ins->setName("RigidBodyGUI");
				ins->setVersion("1.0");
				ins->setDescription("Rigid body GUI extensions for QtGUI");

				gInstance.store(ins, std::memory_order_release);
			}
		}
		return ins;
	}

	void RigidBodyGUIInitializer::initializeActions()
	{
		// Register "Open" and "Create Rigid Body" for common 3D model file types
		std::vector<std::string> modelExts = { "obj", "gltf", "glb", "fbx", "stl" };
		for (const std::string& ext : modelExts)
		{
			QContentBrowser::registerAction(ext, "Open", openMeshCallback);
			QContentBrowser::registerAction(ext, "Create Rigid Body", createRigidBodyCallback);
		}

		QContentBrowser::registerAction("pdm", "Edit", editPDMCallback);

		NodeFactory* factory = NodeFactory::instance();

		factory->addContentAction(std::string("pdm"),
			[=](const std::string& path)->std::shared_ptr<Node>
			{
				auto node = std::make_shared<ConfigurableBody<DataType3f>>();
				node->varLoadConfigPath()->setValue(path);
				return node;
			});

	}

	/**
	 * "Open" action: load the mesh file and show it in a render window
	 */
	static void openMeshCallback(const QString& filePath, QWidget* parent)
	{
		QFileInfo fileInfo(filePath);
		std::string ext = fileInfo.suffix().toStdString();

		auto ext2Act = NodeFactory::instance()->nodeContentActions();
		if (ext2Act.find(ext) == ext2Act.end())
		{
			QMessageBox::warning(parent, "Warning",
				"No loader registered for file type: " + fileInfo.suffix());
			return;
		}

		auto func = ext2Act[ext];
		if (!func)
			return;

		auto node = func(filePath.toStdString());
		if (!node)
			return;

		RigidBodyGUIInitializer::assetNodes.push_back(node);
		auto texMeshInterface = std::dynamic_pointer_cast<TextureMeshInterface>(node);
		if (!texMeshInterface)
		{
			QMessageBox::warning(parent, "Warning", "This file type does not support TextureMesh preview.");
			return;
		}

		auto field = texMeshInterface->getTextureMesh();
		if (!field)
			return;

		QWidget* renderWindow = new QWidget();
		renderWindow->setWindowTitle("Mesh Viewer - " + fileInfo.fileName());
		renderWindow->setMinimumSize(800, 600);

		QVBoxLayout* layout = new QVBoxLayout(renderWindow);
		layout->setContentsMargins(0, 0, 0, 0);

		PTextureMeshViewerWidget* viewer = new PTextureMeshViewerWidget(field, renderWindow);
		layout->addWidget(viewer);

		renderWindow->setLayout(layout);
		renderWindow->setAttribute(Qt::WA_DeleteOnClose);
		renderWindow->show();
	}

	
	/**
	 * "Create Rigid Body" action: show a config dialog and generate a .pdm file
	 */
	static void createRigidBodyCallback(const QString& filePath, QWidget* parent)
	{
		QFileInfo fileInfo(filePath);

		// --- Configuration Window ---
		QWidget* dialog = new QWidget(parent);
		dialog->setWindowTitle("Create Rigid Body Configuration: " + fileInfo.fileName());
		dialog->setMinimumSize(1280, 720);
		dialog->setAttribute(Qt::WA_DeleteOnClose);
		dialog->setWindowFlag(Qt::Window);

		QVBoxLayout* mainLayout = new QVBoxLayout(dialog);
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);

		QWidget* toolBar = new QWidget;
		toolBar->setFixedHeight(76);
		toolBar->setStyleSheet(R"(
				border: 1px solid #464646;
				border-radius: 4px;
				padding: 4px 8px;
		)");
		QHBoxLayout* toolBarLayout = new QHBoxLayout(toolBar);
		toolBarLayout->setContentsMargins(8, 4, 8, 4);
		toolBarLayout->setSpacing(6);

		QToolButton* saveBtn = new QToolButton();
		saveBtn->setFixedHeight(76);
		saveBtn->setFixedWidth(85);

		saveBtn->setIcon(QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/Save.png")));
		saveBtn->setIconSize(QSize(48, 48));
		saveBtn->setText("Save");
		saveBtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
		toolBarLayout->addWidget(saveBtn);

		QToolButton* saveAsBtn = new QToolButton();
		saveAsBtn->setFixedHeight(76);
		saveAsBtn->setFixedWidth(85);

		saveAsBtn->setIcon(QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/SaveAs.png")));
		saveAsBtn->setIconSize(QSize(48, 48));
		saveAsBtn->setText("Save As");
		saveAsBtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
		toolBarLayout->addWidget(saveAsBtn);

		toolBarLayout->addStretch();

		mainLayout->addWidget(toolBar);

		QSplitter* splitter = new QSplitter(Qt::Horizontal);
		splitter->setHandleWidth(6);
		splitter->setChildrenCollapsible(false);

		QWidget* sideWidget = new QWidget;
		sideWidget->setMinimumWidth(200);
		QVBoxLayout* sideLayout = new QVBoxLayout(sideWidget);
		sideLayout->setContentsMargins(4, 4, 4, 4);

		QLabel* fileLabel = new QLabel("Source file: " + filePath);
		sideLayout->addWidget(fileLabel);

		PPropertyWidget* propertyWidget = new PPropertyWidget();
		propertyWidget->setMinimumWidth(100);

		auto renderWidget = new GLMeshRenderWidget();
		renderWidget->setMinimumSize(400, 300);

		splitter->addWidget(sideWidget);
		splitter->addWidget(renderWidget);
		splitter->setStretchFactor(0, 0);
		splitter->setStretchFactor(1, 1);
		splitter->setSizes({ 1290/3, 1280/5 });

		mainLayout->addWidget(splitter, 1);

		auto configNode = std::make_shared<ConfigurableBody<DataType3f>>();
		QObject::connect(propertyWidget, &PPropertyWidget::nodeUpdated, renderWidget, &GLMeshRenderWidget::onNodeUpdated);
		configNode->varFilePath()->setValue(filePath.toStdString());
		configNode->varVehiclesTransform()->clear();
		configNode->varVehiclesTransform()->pushBack(Transform3f());

		QObject::connect(saveBtn, &QToolButton::released,
			[=]() {
				QString savePath = fileInfo.absolutePath() + "/" + fileInfo.completeBaseName() + ".pdm";
				savePath = QDir::toNativeSeparators(savePath);
				configNode->saveToPath(savePath.toStdString());
				printf("[Save PDM] %s\n", savePath.toStdString().c_str());
			}
		);

		QObject::connect(saveAsBtn, &QToolButton::released,
			[=]() {
				QString defaultPath = fileInfo.absolutePath() + "/" + fileInfo.completeBaseName() + ".pdm";
				QString savePath = QFileDialog::getSaveFileName(dialog, "Save As ...", defaultPath, "Peridyno Multibody Files (*.pdm)");
				if (!savePath.isEmpty()) {
					savePath = QDir::toNativeSeparators(savePath);
					configNode->saveToPath(savePath.toStdString());
					printf("[Save As PDM] %s\n", savePath.toStdString().c_str());
				}
			}
		);

		auto ext = fileInfo.suffix().toStdString();
		std::shared_ptr<Node> importNode;

		if (ext == std::string("fbx"))
		{
			auto fbxNode = std::make_shared<FBXLoader<DataType3f>>();
			importNode = fbxNode;
			fbxNode->varFileName()->setValue(filePath.toStdString());
			fbxNode->stateTextureMesh()->connect(configNode->inTextureMesh());
			importNode->setVisible(false);
		}

		// Add node to scene for processing
		renderWidget->addNode(configNode);
		renderWidget->addNode(importNode);
		
		auto repaintWidget = [=]() {
			renderWidget->update();
		};

		propertyWidget->showNodeProperty(configNode);
		sideLayout->addWidget(propertyWidget, 1);

		QObject::connect(propertyWidget, QOverload<std::shared_ptr<Node>>::of(&PPropertyWidget::nodeUpdated), repaintWidget);
		propertyWidget->getPropertyWidget(1)->setVisible(false);
		propertyWidget->getLockerButton(0)->setVisible(false);
		propertyWidget->getLockerButton(1)->setVisible(false);
	
		QPushButton* resetScene = new QPushButton("Reset Scene");

		sideLayout->addStretch();
		sideLayout->addWidget(resetScene);
		auto conn = QObject::connect(resetScene, &QPushButton::released,
			[=]() {

				renderWidget->resetScene();
				renderWidget->update();
				printf("[Reset RigidBodyGUI Scene]\n");
			}
		);

		renderWidget->resetScene();
		renderWidget->update();

		dialog->show();
	}

	static void editPDMCallback(const QString& filePath, QWidget* parent) 
	{
		QFileInfo fileInfo(filePath);

		// --- Configuration Window ---
		QWidget* dialog = new QWidget(parent);
		dialog->setWindowTitle("Edit pdm file: " + fileInfo.fileName());
		dialog->setMinimumSize(1280, 720);
		dialog->setAttribute(Qt::WA_DeleteOnClose);
		dialog->setWindowFlag(Qt::Window);

		QVBoxLayout* mainLayout = new QVBoxLayout(dialog);
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);

		QWidget* toolBar = new QWidget;
		toolBar->setFixedHeight(82);
		toolBar->setStyleSheet(R"(
				border: 1px solid #464646;
				border-radius: 4px;
				padding: 4px 8px;
		)");
		QHBoxLayout* toolBarLayout = new QHBoxLayout(toolBar);
		toolBarLayout->setContentsMargins(8, 4, 8, 4);
		toolBarLayout->setSpacing(6);

		QToolButton* saveBtn = new QToolButton();
		saveBtn->setFixedHeight(76);
		saveBtn->setFixedWidth(85);
		

		saveBtn->setIcon(QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/Save.png")));
		saveBtn->setIconSize(QSize(48, 48));
		saveBtn->setText("Save");
		saveBtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
		toolBarLayout->addWidget(saveBtn);

		QToolButton* saveAsBtn = new QToolButton();
		saveAsBtn->setFixedHeight(76);
		saveAsBtn->setFixedWidth(85);

		saveAsBtn->setIcon(QIcon(QString::fromStdString(getAssetPath() + "icon/ToolBarIco/File/SaveAs.png")));
		saveAsBtn->setIconSize(QSize(48, 48));
		saveAsBtn->setText("Save As");
		saveAsBtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
		toolBarLayout->addWidget(saveAsBtn);

		toolBarLayout->addStretch();

		mainLayout->addWidget(toolBar);

		QSplitter* splitter = new QSplitter(Qt::Horizontal);
		splitter->setHandleWidth(6);
		splitter->setChildrenCollapsible(false);

		QWidget* sideWidget = new QWidget;
		sideWidget->setMinimumWidth(200);
		QVBoxLayout* sideLayout = new QVBoxLayout(sideWidget);
		sideLayout->setContentsMargins(4, 4, 4, 4);

		QLabel* fileLabel = new QLabel("Source file: " + filePath);
		sideLayout->addWidget(fileLabel);

		PPropertyWidget* propertyWidget = new PPropertyWidget();
		propertyWidget->setMinimumWidth(150);

		auto renderWidget = new GLMeshRenderWidget();
		renderWidget->setMinimumSize(400, 300);

		splitter->addWidget(sideWidget);
		splitter->addWidget(renderWidget);
		splitter->setStretchFactor(0, 0);
		splitter->setStretchFactor(1, 1);
		splitter->setSizes({ 1290 / 2, 1280 / 2 });

		mainLayout->addWidget(splitter, 1);

		auto configNode = std::make_shared<ConfigurableBody<DataType3f>>();
		renderWidget->addNode(configNode);

		QObject::connect(propertyWidget, &PPropertyWidget::nodeUpdated, renderWidget, &GLMeshRenderWidget::onNodeUpdated);
		configNode->varLoadConfigPath()->setValue(FilePath(filePath.toStdString()));

		QObject::connect(saveBtn, &QToolButton::released,
			[=]() {
				std::string savePath = configNode->varLoadConfigPath()->getValue().string();
				configNode->saveToPath(savePath);
				printf("[Save PDM] %s\n", savePath.c_str());
			}
		);

		QObject::connect(saveAsBtn, &QToolButton::released,
			[=]() {
				QString currentPath = QString::fromStdString(configNode->varLoadConfigPath()->getValue().string());
				QString savePath = QFileDialog::getSaveFileName(dialog, "Save As ...", currentPath, "Peridyno Multibody Files (*.pdm)");
				if (!savePath.isEmpty()) {
					savePath = QDir::toNativeSeparators(savePath);
					configNode->saveToPath(savePath.toStdString());
					configNode->varLoadConfigPath()->setValue(FilePath(savePath.toStdString()));
					dialog->setWindowTitle("Edit pdm file: " + QFileInfo(savePath).fileName());
					printf("[Save As PDM] %s\n", savePath.toStdString().c_str());
				}
			}
		);

		FilePath importFilePath = configNode->varFilePath()->getValue();
		auto ext = importFilePath.path().extension().string();
		std::shared_ptr<Node> importNode;

		if (ext == std::string(".fbx"))
		{
			auto fbxNode = std::make_shared<FBXLoader<DataType3f>>();
			importNode = fbxNode;
			fbxNode->varFileName()->setValue(importFilePath);
			fbxNode->stateTextureMesh()->connect(configNode->inTextureMesh());
			importNode->setVisible(false);
		}

		renderWidget->addNode(importNode);

		auto repaintWidget = [=]() {
			renderWidget->update();
		};

		propertyWidget->showNodeProperty(configNode);
		sideLayout->addWidget(propertyWidget, 1);

		QObject::connect(propertyWidget, QOverload<std::shared_ptr<Node>>::of(&PPropertyWidget::nodeUpdated), repaintWidget);
		propertyWidget->getPropertyWidget(1)->setVisible(false);
		propertyWidget->getLockerButton(0)->setVisible(false);
		propertyWidget->getLockerButton(1)->setVisible(false);

		QPushButton* resetScene = new QPushButton("Reset Scene");

		sideLayout->addStretch();
		sideLayout->addWidget(resetScene);
		auto conn = QObject::connect(resetScene, &QPushButton::released,
			[=]() {

				renderWidget->resetScene();
				renderWidget->update();
				printf("[Reset RigidBodyGUI Scene]\n");
			}
		);

		renderWidget->resetScene();
		renderWidget->update();

		dialog->show();
	}
}

// Static plugin initialization (called when statically linked)
dyno::PluginEntry* RigidBodyGUI::initStaticPlugin()
{
	if (dyno::RigidBodyGUIInitializer::instance()->initialize())
		return dyno::RigidBodyGUIInitializer::instance();

	return nullptr;
}

// Dynamic plugin entry point (called by PluginManager::loadPlugin)
PERIDYNO_API dyno::PluginEntry* RigidBodyGUI::initDynoPlugin()
{
	if (dyno::RigidBodyGUIInitializer::instance()->initialize())
		return dyno::RigidBodyGUIInitializer::instance();

	return nullptr;
}

// Auto-initialization: this static object ensures registration happens when DLL is loaded
namespace
{
	struct AutoInit
	{
		AutoInit()
		{
			// Automatically initialize when DLL loads
			RigidBodyGUI::initStaticPlugin();
		}
	};

	static AutoInit gAutoInit;
}