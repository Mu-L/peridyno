#ifndef QNODEPROPERTYWIDGET_H
#define QNODEPROPERTYWIDGET_H

#include <QToolBox>
#include <QWidget>
#include <QGroupBox>
#include <QScrollArea>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QLabel>

#include "nodes/QNode"
#include "LockerButton.h"

#include <vector>



namespace dyno
{
	class Node;
	class Module;
	class OBase;
	class FBase;
	class QDoubleSpinner;
	class PVTKOpenGLWidget;

	class QBoolFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QBoolFieldWidget(FBase* field);
		~QBoolFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int status);

	private:
		FBase* m_field = nullptr;
	};


	class QFInstanceWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QFInstanceWidget(FBase* field);
		~QFInstanceWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int status);

	private:
		FBase* m_field = nullptr;
	};

	class QIntegerFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QIntegerFieldWidget(FBase* field);
		~QIntegerFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int);

	private:
		FBase* m_field = nullptr;
	};

	class QRealFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QRealFieldWidget(FBase* field);
		~QRealFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		FBase* m_field = nullptr;
	};


	class QVector3FieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QVector3FieldWidget(FBase* field);
		~QVector3FieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		FBase* m_field = nullptr;

		QDoubleSpinner* spinner1;
		QDoubleSpinner* spinner2;
		QDoubleSpinner* spinner3;
	};

	class QStateFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QStateFieldWidget(FBase* field);
		~QStateFieldWidget() {};

	Q_SIGNALS:
		void fieldTypeChanged();

	public slots:
		void tagAsOuput(int status);

	private:
		FBase* m_field = nullptr;
	};

	class PPropertyWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PPropertyWidget(QWidget *parent = nullptr);
		~PPropertyWidget();

		virtual QSize sizeHint() const;

//		void clear();

	//signals:
		QWidget* addWidget(QWidget* widget);
		void removeAllWidgets();

	signals:
		void fieldUpdated();

	public slots:
		void showProperty(Module* module);
		void showProperty(Node* node);

		void showNodeProperty(Qt::QtNode& block);

		void updateDisplay();

	private:
		void updateContext(OBase* base);

		void addScalarFieldWidget(FBase* field, QGridLayout* layout,int j);
		void addArrayFieldWidget(FBase* field);
		void addInstanceFieldWidget(FBase* field);

		void addStateFieldWidget(FBase* field);

		QVBoxLayout* m_main_layout;
		QScrollArea* m_scroll_area;
		QWidget * m_scroll_widget;
		QGridLayout* m_scroll_layout;

		std::vector<QWidget*> m_widgets;


		LockerButton* m_sizeButton;
		LockerButton* m_states;
		LockerButton* m_positionButton;
		QWidget* m_sizeWidget;
		QWidget* m_statesWidget;

		QWidget* m_positionWidget;
		quint8 m_sizeList;
		quint8 m_positionList;

	};

}

#endif // QNODEPROPERTYWIDGET_H
