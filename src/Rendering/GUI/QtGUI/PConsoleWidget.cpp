#include "PConsoleWidget.h"
#include "Platform.h"
#include <QListWidget>
#include <QDir>
#include <QStringList>
#include <QHBoxLayout>
#include <QListView>
#include <QPixmap>
#include "NodeFactory.h"

namespace dyno
{
	class CustomFileSystemModel : public QFileSystemModel
	{
	public:
		explicit CustomFileSystemModel(QObject* parent = nullptr) {};
		~CustomFileSystemModel() {};
	private:
		QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const
		{
			if (index.isValid())
			{
				if (role == Qt::DecorationRole)
				{
					QFileInfo info = CustomFileSystemModel::fileInfo(index);
					if (info.isFile())
					{
						if (info.suffix() == "png" || info.suffix() == "jpg" || info.suffix() == "bmp")
						{
							std::string iconPath = getAssetPath() + "/icon/ContentBrowser/image.png";
							return QPixmap(iconPath.c_str());
						}
						else if (info.suffix() == "obj" || info.suffix() == "gltf" || info.suffix() == "glb" || info.suffix() == "fbx" || info.suffix() == "STL" || info.suffix() == "stl")
						{
							std::string iconPath = getAssetPath() + "/icon/ContentBrowser/3dModel.png";
							return QPixmap(iconPath.c_str());
						}
					}
				}
			}

			return QFileSystemModel::data(index, role);
		}
	};


	PConsoleWidget::PConsoleWidget(QWidget *parent) :
		QWidget(parent)
	{
//		setMinimumHeight(200);
	}

	QContentBrowser::QContentBrowser(QWidget* parent /*= nullptr*/)
		: QWidget()
	{
		QHBoxLayout* layout = new QHBoxLayout(this);
		this->setLayout(layout);

		std::string path = getAssetPath();
		QDir root(path.c_str());

		//Add file browser
		model = new QFileSystemModel(this);
		model->setRootPath(path.c_str());
		model->setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
		model->sort(0, Qt::AscendingOrder);

//		QObject::connect(model, SIGNAL(directoryLoaded(QString)), SLOT(findDirectory(QString)));

		// File system
		treeView = new QTreeView(this);
		treeView->setModel(model);
		treeView->setHeaderHidden(true);	//Show tree header
		treeView->setFixedWidth(520);
		treeView->hideColumn(1);
		treeView->hideColumn(2);			//Hide second column
		treeView->hideColumn(3);
		treeView->setRootIndex(model->index(path.c_str()));
		layout->addWidget(treeView);



		QStringList filter;
		filter <<"*.png" << "*.jpg" << "*.bmp" << "*.obj" << "*.gltf" << "*.glb" << "*.fbx" << "*.STL" << "*.stl" << "*.xml";
		auto* listModel = new CustomFileSystemModel(this);
		listModel->setRootPath(path.c_str());
		listModel->setFilter(QDir::Files | QDir::NoDotAndDotDot);
		listModel->setNameFilters(filter);
		listModel->setNameFilterDisables(false);
		listModel->sort(0, Qt::AscendingOrder);

		listView = new QListView;
		listView->setModel(listModel);
		listView->setViewMode(QListView::IconMode);
		listView->setIconSize(QSize(80, 80));
		listView->setGridSize(QSize(120, 120));
		listView->setUniformItemSizes(true);
		listView->setResizeMode(QListWidget::Adjust);
		listView->setTextElideMode(Qt::ElideRight);
		listView->setRootIndex(listModel->index(path.c_str()));
		layout->addWidget(listView);

		connect(treeView, SIGNAL(clicked(const QModelIndex&)),
			this, SLOT(treeItemSelected(const QModelIndex&)));

		connect(listView, SIGNAL(clicked(const QModelIndex&)),
			this, SLOT(assetItemSelected(const QModelIndex&)));

		connect(listView, SIGNAL(doubleClicked(const QModelIndex&)),
			this, SLOT(assetDoubleClicked(const QModelIndex&)));
	}

	void QContentBrowser::treeItemSelected(const QModelIndex& index)
	{
		QString name = model->fileName(index);
		QString path = model->fileInfo(index).absolutePath() + "/" + name;

		//A hack
		QStringList filter;
		filter << "*.png" << "*.jpg" << "*.bmp" << "*.obj" << "*.gltf" << "*.glb" << "*.fbx" << "*.STL" << "*.stl" << "*.xml";
		auto* newListModel = new CustomFileSystemModel(this);
		newListModel->setRootPath(path);
		newListModel->setFilter(QDir::Files | QDir::NoDotAndDotDot);
		newListModel->setNameFilters(filter);
		newListModel->setNameFilterDisables(false);
		newListModel->sort(0, Qt::AscendingOrder);

 		listView->setModel(newListModel);
		listView->setRootIndex(newListModel->index(path));
	}

	void QContentBrowser::assetItemSelected(const QModelIndex& index)
	{
		QString name = model->fileName(index);
		QString path = model->fileInfo(index).absolutePath() + "/" + name;


	}

	void QContentBrowser::assetDoubleClicked(const QModelIndex& index)
	{
		QString name = model->fileName(index);
		QString path = model->fileInfo(index).absolutePath() + "/" + name;

		std::cout << path.toStdString() << "\n";

		auto ext = model->fileInfo(index).suffix().toStdString();
		auto ext2Act = NodeFactory::instance()->nodeContentActions();
		if (ext2Act.find(ext) != ext2Act.end())
		{
			auto func = ext2Act[ext];
			if (func != nullptr) {
				auto node = func(path.toStdString());

				emit nodeCreated(node);
			}
		}
	}
}