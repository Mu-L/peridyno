#pragma once
#include "ParticleSystem.h"

namespace dyno
{
	template<typename> class NeighborPointQuery;
	template<typename> class PointSetToPointSet;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;
	template<typename> class ElastoplasticityModule;
	template<typename> class DensityPBD;
	template<typename TDataType> class ImplicitViscosity;
	/*!
	*	\class	ParticleElastoplasticBody
	*	\brief	Peridynamics-based elastoplastic object.
	*/
	template<typename TDataType>
	class ParticleElastoplasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleElastoplasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleElastoplasticBody(std::string name = "default");
		virtual ~ParticleElastoplasticBody();

		void advance(Real dt) override;

		void updateTopology() override;

		bool initialize() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void loadSurface(std::string filename);

		void setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver);

		std::shared_ptr<Node> getSurfaceNode() { return m_surfaceNode; }

	public:
		VarField<Real> m_horizon;

	private:
		std::shared_ptr<Node> m_surfaceNode;

		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborPointQuery<TDataType>> m_nbrQuery;
		std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
		std::shared_ptr<ElastoplasticityModule<TDataType>> m_plasticity;
		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;
	};
}