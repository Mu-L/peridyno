#pragma once

#include <Wt/WAbstractItemModel.h>
#include <Wt/WAbstractTableModel.h>
#include <Wt/WText.h>
#include <Wt/WPanel.h>
#include <Wt/WTable.h>
#include <Wt/WDoubleSpinBox.h>
#include <Wt/WLogger.h>

#include <FBase.h>
#include "Field/FilePath.h"

#include "WtGUI/PropertyItem/WRealFieldWidget.h"
#include "WtGUI/PropertyItem/WVector3FieldWidget.h"
#include "WtGUI/PropertyItem/WVector3iFieldWidget.h"
#include "WtGUI/PropertyItem/WBoolFieldWidget.h"
#include "WtGUI/PropertyItem/WIntegerFieldWidget.h"
#include "WtGUI/PropertyItem/WColorWidget.h"
#include "WtGUI/PropertyItem/WFileWidget.h"
#include "WtGUI/PropertyItem/WEnumFieldWidget.h"
#include "WtGUI/PropertyItem/WColorWidget.h"
#include "WtGUI/PropertyItem/WStateFieldWidget.h"

namespace dyno
{
	class Node;
	class Module;
	class SceneGraph;
	class FBase;
};

class WParameterDataNode : public Wt::WAbstractTableModel
{
public:

	WParameterDataNode();
	~WParameterDataNode();

	void setNode(std::shared_ptr<dyno::Node> node);
	void setModule(std::shared_ptr<dyno::Module> module);

	virtual int columnCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;
	virtual int rowCount(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;
	//virtual int rowCountModule(const Wt::WModelIndex& parent = Wt::WModelIndex()) const;

	virtual Wt::cpp17::any data(const Wt::WModelIndex& index,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	virtual Wt::cpp17::any headerData(int section,
		Wt::Orientation orientation = Wt::Orientation::Horizontal,
		Wt::ItemDataRole role = Wt::ItemDataRole::Display) const;

	void createParameterPanel(Wt::WContainerWidget* parameterWidget);
	void createParameterPanelModule(Wt::WPanel* panel);

	void updateNode();
	void updateModule();

	Wt::Signal<int>& changeValue()
	{
		return changeValue_;
	}

	void emit();

public:
	struct FieldWidgetMeta {
		using constructor_t = Wt::WContainerWidget* (*)(dyno::FBase*);
		const std::type_info* type;
		constructor_t constructor;
	};

	static int registerWidget(const FieldWidgetMeta& meta);

	static FieldWidgetMeta* getRegistedWidget(const std::string&);

	Wt::WContainerWidget* createFieldWidget(dyno::FBase* field);

private:

	std::shared_ptr<dyno::Node> mNode;
	std::shared_ptr<dyno::Module> mModule;
	Wt::Signal<int> changeValue_;

	void castToDerived(Wt::WContainerWidget* fw);

	void addScalarFieldWidget(Wt::WTable* table, std::string label, dyno::FBase* field, int labelWidth = 150, int widgetWidth = 300);

	void addStateFieldWidget(Wt::WTable* table, dyno::FBase* field);

	static std::map<std::string, FieldWidgetMeta> sFieldWidgetMeta;
};
