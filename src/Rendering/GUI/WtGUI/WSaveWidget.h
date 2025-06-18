#pragma once
#include "SceneGraph.h"

#include <Wt/Http/Request.h>
#include <Wt/Http/Response.h>
#include <Wt/WContainerWidget.h>
#include <Wt/WFileResource.h>
#include <Wt/WFileUpload.h>
#include <Wt/WResource.h>
#include <Wt/WVBoxLayout.h>

namespace dyno
{
	class SceneLoaderFactory;
	class SceneGraphFactory;

}

class WMainWindow;

class WSaveWidget : public Wt::WContainerWidget
{
public:
	WSaveWidget(WMainWindow* parent, int width);
	~WSaveWidget();

private:

	void createSavePanel();

	void createUploadPanel();

	void save(std::string fileName);

	void recreate();

	bool isValidFileName(const std::string& filename);

	std::string uploadFile(Wt::WFileUpload* upload);

	std::string removeXmlExtension(const std::string& filename);

private:

	Wt::WText* mSaveOut;
	Wt::WText* mUploadOut;

	Wt::WVBoxLayout* mSaveLayout;
	WMainWindow* mParent;

	std::shared_ptr<dyno::SceneGraph> mScene;
	int mWidth;
};

class downloadResource : public Wt::WFileResource
{
public:
	downloadResource(std::string fileName) : Wt::WFileResource()
	{
		suggestFileName(fileName);
		filePath = fileName;
	}

	~downloadResource()
	{
		beingDeleted();
	}

	void handleRequest(const Wt::Http::Request& request,
		Wt::Http::Response& response) {
		response.setMimeType("text/xml");


		std::ifstream inputFile(filePath);
		if (!inputFile) {
			std::cout << "�޷��������ļ�" << std::endl;
			return;
		}

		// ��ȡ�����ļ���д������ļ�
		std::string line, lines;
		while (std::getline(inputFile, line)) {
			lines = lines + line + "\n";
		}
		inputFile.close();

		response.out() << lines;
	}


private:
	std::string filePath;
};