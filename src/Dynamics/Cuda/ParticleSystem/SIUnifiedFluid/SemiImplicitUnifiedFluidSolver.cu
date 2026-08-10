#include "SemiImplicitUnifiedFluidSolver.h"
#include "ParticleSystem/Module/SummationDensity.h"
#include "Algorithm/Reduction.h"
#include "Algorithm/Arithmetic.h"
#include "Matrix/MatrixFunc.h"

namespace dyno
{

	template<typename TDataType>
	SemiImplicitUnifiedFluidSolver<TDataType>::SemiImplicitUnifiedFluidSolver()
		: ParticleApproximation<TDataType>()
	{
		this->varRestDensity()->setValue(Real(1000));
		m_summation = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->quote(m_summation->varRestDensity());
		this->inSmoothingLength()->connect(m_summation->inSmoothingLength());
		this->inSamplingDistance()->connect(m_summation->inSamplingDistance());
		this->inPosition()->connect(m_summation->inPosition());
		this->inNeighborIds()->connect(m_summation->inNeighborIds());
		m_summation->outDensity()->connect(this->outDensity());
		this->varKernelType()->getDataPtr()->setCurrentKey(2);

		this->inAttribute()->tagOptional(true);
	}

	template<typename TDataType>
	SemiImplicitUnifiedFluidSolver<TDataType>::~SemiImplicitUnifiedFluidSolver()
	{
		m_Source.clear();
		m_matK.clear();
		mY.clear();
		mYStar.clear();
	}


	template<typename Real>
	__device__ Real  K_SIUFS_SimpleAdsorptionFunc(Real r, Real r0)
	{
		Real q = r / r0;
		Real value = 1 - (q - 2) * (q - 2);
		if (value < 0.0f)
			value = 0.0f;
		return value;
	}


	template<typename Real>
	__device__ Real  K_SIUFS_EnergyForSimpleAdsorptionFunc(Real r, Real r0)
	{
		Real q = r / r0;
		Real temp = (q - 2);
		Real value = q - temp * temp * temp / (3.0) - 4.0 / 3.0;
		return value;
	}

	template<typename Real>
	__device__ Real  K_SIUFS_SimpleSimpleBidirectionFuncPositive(Real r, Real r0)
	{
		Real q = r / r0;
		Real value = 0.0f;
		if (q <= 1)
		{
			value = q * q;
		}
		else if (q <= 3)
		{
			value = 1 - (q - 2) * (q - 2);
		}
		else
		{
			value = 0.0f;
		}
		return value;
	}

	template<typename Real>
	__device__ Real  K_SIUFS_SimpleSimpleBidirectionFuncNegative(Real r, Real r0)
	{
		Real q = r / r0;
		Real value = 0.0f;
		if (q <= 1)
		{
			value = -1;
		}
		else
		{
			value = 0.0f;
		}
		return value;
	}

	template<typename Real>
	__device__ Real  K_SIUFS_EnergyForSimpleBidirectionFuncNegative(Real r, Real r0)
	{
		Real q = r / r0;
		Real Energy = 0.0f;

		if (q <= 1)
		{
			Energy = (q * q * q / 3) - q + 2.0 / 3.0;
		}
		else if (q <= 3)
		{
			Energy = q - (q - 2) * (q - 2) * (q - 2) / 3 - 4.0 / 3.0;
		}
		return Energy;
	}

	template<typename Real, typename Coord, typename Matrix>
	__global__ void SIUFS_ComputeSourceForSurfaceTension(
		DArray<Matrix> mK,
		DArray<Coord> Sources,
		DArray<Coord> y,
		DArray<Coord> y_star,
		DArrayList<int> NeighborLists,
		Real strength,
		Real r0,
		Real h,
		Real dt,
		int pairWiseFunc
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Sources.size()) return;

		Real cij_positive(0.0f);
		Real cij_negative(0.0f);

		List<int>& list_i = NeighborLists[pId];
		int nbSize = list_i.size();

		Coord total_s(0.0f);
		Real total_r(0.0f);
		Real b = strength * dt * dt;

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord yij = y[pId] - y[j];
			Real rij = yij.norm();
			if (rij > EPSILON)
			{

				if (pairWiseFunc == 0)
				{
					cij_positive = K_SIUFS_SimpleAdsorptionFunc(rij, r0);
				}
				else if (pairWiseFunc == 1)
				{
					cij_positive = K_SIUFS_SimpleSimpleBidirectionFuncPositive(rij, r0);
					cij_negative = K_SIUFS_SimpleSimpleBidirectionFuncNegative(rij, r0);
				}

				Real tp_ij = b * (cij_positive / rij);

				Coord s_ij = tp_ij * y[j] + b * (cij_negative / rij) * (y[j] - y[pId]);
				Coord s_ji = tp_ij * y[pId] + b * (cij_negative / rij) * (y[pId] - y[j]);

				total_s += s_ij;
				total_r += tp_ij;
				
				atomicAdd(&Sources[j][0], s_ji[0]);
				atomicAdd(&Sources[j][1], s_ji[1]);
				atomicAdd(&Sources[j][2], s_ji[2]);
				atomicAdd(&mK[j](0, 0), tp_ij);
				atomicAdd(&mK[j](1, 1), tp_ij);
				atomicAdd(&mK[j](2, 2), tp_ij);
			}
		}
		atomicAdd(&Sources[pId][0], total_s[0]);
		atomicAdd(&Sources[pId][1], total_s[1]);
		atomicAdd(&Sources[pId][2], total_s[2]);
		atomicAdd(&mK[pId](0, 0), total_r);
		atomicAdd(&mK[pId](1, 1), total_r);
		atomicAdd(&mK[pId](2, 2), total_r);
	}

	template<typename Real, typename Coord, typename Matrix>
	__global__ void SIUFS_ComputeSourceForSurfaceTension_GhostSolidBoundary(
		DArray<Matrix> mK,
		DArray<Coord> Sources,
		DArray<Coord> y,
		DArray<Coord> y_star,
		DArrayList<int> NeighborLists,
		DArray<Attribute> Attributes,
		Real strength_solid,
		Real strength_fluid,
		Real r0,
		Real h,
		Real dt,
		int pairWiseFunc
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Sources.size()) return;
		if (Attributes[pId].isFixed())
		{
			return;
		}

		Real cij_positive(0.0f);
		Real cij_negative(0.0f);

		List<int>& list_i = NeighborLists[pId];
		int nbSize = list_i.size();

		Coord total_s(0.0f);
		Real total_r(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord yij = y[pId] - y[j];
			Real rij = yij.norm();
			if (rij > EPSILON)
			{
				if (pairWiseFunc == 0)
				{
					cij_positive = K_SIUFS_SimpleAdsorptionFunc(rij, r0);
				}
				else if (pairWiseFunc == 1)
				{
					cij_positive = K_SIUFS_SimpleSimpleBidirectionFuncPositive(rij, r0);
					cij_negative = K_SIUFS_SimpleSimpleBidirectionFuncNegative(rij, r0);
				}

				Real strength_j = 0.0;
				if (Attributes[j].isFixed())
				{
					strength_j = strength_solid;
				}
				else
				{
					strength_j = strength_fluid;
				}

				Real tp_ij = strength_j * dt * dt * (cij_positive / rij);
				Coord s_ij = tp_ij * y[j] + strength_j * dt * dt * (cij_negative / rij) * (y[j] - y[pId]);
				Coord s_ji = tp_ij * y[pId] + strength_j * dt * dt * (cij_negative / rij) * (y[pId] - y[j]);

				total_s += s_ij;
				total_r += tp_ij;

				atomicAdd(&Sources[j][0], s_ji[0]);
				atomicAdd(&Sources[j][1], s_ji[1]);
				atomicAdd(&Sources[j][2], s_ji[2]);
				atomicAdd(&mK[j](0, 0), tp_ij);
				atomicAdd(&mK[j](1, 1), tp_ij);
				atomicAdd(&mK[j](2, 2), tp_ij);
			}
		}

		atomicAdd(&Sources[pId][0], total_s[0]);
		atomicAdd(&Sources[pId][1], total_s[1]);
		atomicAdd(&Sources[pId][2], total_s[2]);
		atomicAdd(&mK[pId](0, 0), total_r);
		atomicAdd(&mK[pId](1, 1), total_r);
		atomicAdd(&mK[pId](2, 2), total_r);
	}

	template<typename Real, typename Coord, typename Matrix, typename Kernel>
	__global__ void SIUFS_CalculateSourceForViscosity(
		DArray<Matrix> matK,
		DArray<Coord> src,
		DArray<Coord> Y,
		DArray<Coord> Y_star,
		DArray<Coord> X,
		DArrayList<int> neighbors,
		Real rho_0,
		Real lambda,
		Real mu,
		Real smoothingLength,
		Real dt,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;

		Real alpha = lambda * dt / rho_0;
		Real beta = mu * dt / rho_0;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		
		Coord X_i = X[pId];
		Coord Y_i = Y[pId];

		Coord s_i = Coord(0);
		Matrix matK_i(0);

		Real t_E(0.0);

		Real total_weight(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord X_j = X[j];

			Real rij = (X[pId] - X_j).norm();

			if (rij > EPSILON)
			{
				Real w_ij = weight(rij, smoothingLength, scale);

				Coord Y_j = Y[j];
				Coord nij = (X_j - X_i) / rij;

				Real w_alpha = alpha * w_ij;
				Matrix K_ij_alpha(0);
				K_ij_alpha(0, 0) += nij[0] * nij[0]; K_ij_alpha(0, 1) += nij[0] * nij[1]; K_ij_alpha(0, 2) += nij[0] * nij[2];
				K_ij_alpha(1, 0) += nij[1] * nij[0]; K_ij_alpha(1, 1) += nij[1] * nij[1]; K_ij_alpha(1, 2) += nij[1] * nij[2];
				K_ij_alpha(2, 0) += nij[2] * nij[0]; K_ij_alpha(2, 1) += nij[2] * nij[1]; K_ij_alpha(2, 2) += nij[2] * nij[2];

				Real w_beta = beta * w_ij;
				Matrix K_ij_beta = Matrix::identityMatrix() - K_ij_alpha;

				Coord s_ij = w_alpha * K_ij_alpha * Y_j + w_beta * K_ij_beta * Y_j - w_alpha * K_ij_alpha * (X_j - X_i) - w_beta * K_ij_beta * (X_j - X_i);
				Coord s_ji = w_alpha * K_ij_alpha * Y_i + w_beta * K_ij_beta * Y_i - w_alpha * K_ij_alpha * (X_i - X_j) - w_beta * K_ij_beta * (X_i - X_j);

				t_E += nij.dot((Y_i - Y_j) - (X_i - X_j)) * w_ij / dt;

				total_weight += w_ij;

				atomicAdd(&src[j][0], s_ji[0]);
				atomicAdd(&src[j][1], s_ji[1]);
				atomicAdd(&src[j][2], s_ji[2]);

				s_i += s_ij;

				Matrix K_ij = w_alpha * K_ij_alpha + w_beta * K_ij_beta;
				atomicAdd(&matK[j](0, 0), K_ij(0, 0));
				atomicAdd(&matK[j](0, 1), K_ij(0, 1));
				atomicAdd(&matK[j](0, 2), K_ij(0, 2));
				atomicAdd(&matK[j](1, 0), K_ij(1, 0));
				atomicAdd(&matK[j](1, 1), K_ij(1, 1));
				atomicAdd(&matK[j](1, 2), K_ij(1, 2));
				atomicAdd(&matK[j](2, 0), K_ij(2, 0));
				atomicAdd(&matK[j](2, 1), K_ij(2, 1));
				atomicAdd(&matK[j](2, 2), K_ij(2, 2));
 
 				matK_i += K_ij;
			}
		}

		atomicAdd(&src[pId][0], s_i[0]);
		atomicAdd(&src[pId][1], s_i[1]);
		atomicAdd(&src[pId][2], s_i[2]);
		atomicAdd(&matK[pId](0, 0), matK_i(0, 0));
		atomicAdd(&matK[pId](0, 1), matK_i(0, 1));
		atomicAdd(&matK[pId](0, 2), matK_i(0, 2));
		atomicAdd(&matK[pId](1, 0), matK_i(1, 0));
		atomicAdd(&matK[pId](1, 1), matK_i(1, 1));
		atomicAdd(&matK[pId](1, 2), matK_i(1, 2));
		atomicAdd(&matK[pId](2, 0), matK_i(2, 0));
		atomicAdd(&matK[pId](2, 1), matK_i(2, 1));
		atomicAdd(&matK[pId](2, 2), matK_i(2, 2));
	}

	template<typename Real, typename Coord, typename Matrix, typename Kernel>
	__global__ void SIUFS_CalculateSourceForIncompressibility(
		DArray<Matrix> K,
		DArray<Coord> src,
		DArray<Coord> Y,
		DArray<Coord> Y_star,
		DArray<Coord> X,
		DArray<Real> rho,
		DArrayList<int> neighbors,
		Real rho_0,
		Real kappa,
		Real smoothingLength,
		Real dx,
		Real dt,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;

		Real rho_i = rho[pId];
		rho_i = rho_i > rho_0 ? rho_i : rho_0;

		Real A = kappa * dt * dt / rho_0;
		Real C_plus = rho_i / rho_0;
		Real C_minus = Real(-1);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		Matrix mat_t(0.0);

		Coord Y_i = Y[pId];
		Real A_i = 0;
		Coord s_i = Coord(0);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord Y_j = Y[j];

			Real r = (Y_i - Y_j).norm();
			if (r > EPSILON)
			{
				Real a_ij = A * gradient(r, smoothingLength, scale) * (1.0f / r);
				Coord s_ij = C_minus * a_ij * Y_j + C_plus * a_ij * (Y_j - Y_i);
				Coord s_ji = C_minus * a_ij * Y_i + C_plus * a_ij * (Y_i - Y_j);
				Real A_ij = C_minus * a_ij;

				atomicAdd(&src[j][0], s_ji[0]);
				atomicAdd(&src[j][1], s_ji[1]);
				atomicAdd(&src[j][2], s_ji[2]);
				atomicAdd(&K[j](0, 0), A_ij);
				atomicAdd(&K[j](1, 1), A_ij);
				atomicAdd(&K[j](2, 2), A_ij);

				s_i += s_ij;
				A_i += A_ij;
			}
		}
		const Real m = rho_0 * dx * dx * dx;
		const Real C = m / (dt * dt);
		const Real lambda = rho_i / rho_0;

		atomicAdd(&src[pId][0], s_i[0]);
		atomicAdd(&src[pId][1], s_i[1]);
		atomicAdd(&src[pId][2], s_i[2]);
		atomicAdd(&K[pId](0, 0), A_i);
		atomicAdd(&K[pId](1, 1), A_i);
		atomicAdd(&K[pId](2, 2), A_i);
	}

	template<typename Real, typename Coord>
	__global__ void SIUFS_PredictVelocity(
		DArray<Coord> Ystar,
		DArray<Coord> X,
		DArray<Coord> velocity,
		Coord gravity,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Ystar.size()) return;

		Coord vel_star = velocity[pId] + gravity * dt;
		Ystar[pId] = X[pId] + vel_star * dt;
	}

	template<typename Coord, typename Matrix>
	__global__ void SIUFS_CalculateNewPosition(
		DArray<Coord> PositionNext,
		DArray<Coord> Y,		
		DArray<Coord> Y_star,
		DArray<Coord> X,
		DArray<Matrix> A,
		DArray<Coord> src,
		DArray<Real> density,
		Real restDensity,
		Real dt,
		Real m
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;

		Coord posNext_i = (Matrix::identityMatrix() + A[pId]).inverse() * (Y_star[pId] + src[pId]);
		PositionNext[pId] = posNext_i;
	}

	template<typename Real, typename Coord>
	__global__ void SIUFS_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> Y,
		DArray<Coord> X,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;

		vel[pId] = (Y[pId] - X[pId]) / h;
	}

	template<typename Coord, typename Matrix>
	__global__ void SIUFS_CalculateNewPosition(
		DArray<Coord> Y,
		DArray<Coord> Y_star,
		DArray<Coord> X,
		DArray<Attribute> att,
		DArray<Matrix> A,
		DArray<Coord> src,
        Real m
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;
		if (att[pId].isDynamic())
		{
			Y[pId] = (Matrix::identityMatrix() + A[pId]).inverse() * (Y_star[pId] + src[pId]);
		}
		else
		{
			Y[pId] = X[pId];
		}
	}

	template<typename Real, typename Coord>
	__global__ void SIUFS_UpdateVelocity(
		DArray<Coord> vel,
		DArray<Coord> Y,
		DArray<Coord> X,
		DArray<Attribute> att,
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;
		if (att[pId].isFixed())
		{
			vel[pId] = Coord(0.0f);
		
		}
		else {
			vel[pId] = (Y[pId] - X[pId]) / h;
		}
	}

	template<typename TDataType>
	void SemiImplicitUnifiedFluidSolver<TDataType>::resizeArrays() {
		int num = this->inPosition()->size();

		m_matK.resize(num);
		m_matK.reset();

		m_Source.resize(num);
		m_Source.reset();

		mY.resize(num);
		mYStar.resize(num);
		mX.resize(num);
	}

	template<typename TDataType>
	void SemiImplicitUnifiedFluidSolver<TDataType>::compute()
	{
		std::cout << "SemiImplicitUnifiedFluidSolver" << std::endl;

		this->resizeArrays();

		int num = this->inPosition()->size();

		Coord gravity = this->varGravity()->getValue();

		mX.assign(this->inPosition()->getData());
		mYStar.assign(this->inPosition()->constData());

		cuExecute(num,
			SIUFS_PredictVelocity,
			mYStar,
            mX,
			this->inVelocity()->constData(),
			gravity,
			this->inTimeStep()->getValue());

		mY.assign(mYStar);

		auto& mPosNext = this->inPosition()->getData();
		Real dx = this->inSamplingDistance()->getValue();
		Real mass = this->varRestDensity()->getValue() * dx * dx * dx;

		m_summation->varRestDensity()->setValue(this->varRestDensity()->getValue());
		m_summation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());

		/*
		*@brief Iteration begin.
		*/
		for (int i = 0; i < this->varMaxIterationNumber()->getValue(); i++)
		{
			m_summation->update();

			m_Source.reset();
			m_matK.reset();

			if (!this->varDensityConstraintDisable()->getValue())
			{
				/*
				* @note the incompressibility function should be the first.
				*/
				cuFirstOrder(num, this->varKernelType()->currentKey(), this->mScalingFactor,
					SIUFS_CalculateSourceForIncompressibility,
					m_matK,
					m_Source,
					mY,
					mYStar,
					mX,
					m_summation->outDensity()->getData(),
					this->inNeighborIds()->getData(),
					this->varRestDensity()->getValue(),
					this->varKappa()->getValue(),
					this->inSmoothingLength()->getValue(),
					dx,
					this->inTimeStep()->getValue());
			}
			if (!this->varViscosityDisable()->getValue())
			{
				cuZerothOrder(num, this->varKernelType()->currentKey(), this->mScalingFactor,
					SIUFS_CalculateSourceForViscosity,
					m_matK,
					m_Source,
					mY,
					mYStar,
					mX,
					this->inNeighborIds()->getData(),
					this->varRestDensity()->getValue(),
					this->varLambda()->getValue(),
					this->varMu()->getValue(),
					this->inSmoothingLength()->getValue(),
					this->inTimeStep()->getValue());
			}
			if (!this->varSurfaceTensionDisable()->getValue())
			{
				if (this->inAttribute()->size() != num)
				{
					cuExecute(num, SIUFS_ComputeSourceForSurfaceTension,
						m_matK,
						m_Source,
						mY,
						mYStar,
						this->inNeighborIds()->getData(),
						this->varGamma()->getValue(),
						dx,
						this->inSmoothingLength()->getValue(),
						this->inTimeStep()->getValue(),
						this->varPairwiseFunc()->getDataPtr()->currentKey()
					);
				}
				else
				{
					cuExecute(num, SIUFS_ComputeSourceForSurfaceTension_GhostSolidBoundary,
						m_matK,
						m_Source,
						mY,
						mYStar,
						this->inNeighborIds()->getData(),
						this->inAttribute()->getData(),
						this->varSolidAdhension()->getData(),
						this->varGamma()->getValue(),
						dx,
						this->inSmoothingLength()->getValue(),
						this->inTimeStep()->getValue(),
						this->varPairwiseFunc()->getDataPtr()->currentKey()
					);
				}
			}

			if (this->inAttribute()->size() != num)
			{
				cuExecute(num,
					SIUFS_CalculateNewPosition,
					mPosNext,
					mY,
					mYStar,
					mX,
					m_matK,
					m_Source,
					this->outDensity()->getData(),
					this->varRestDensity()->getValue(),
					this->inTimeStep()->getValue(),
					mass
				);
			}
			else
			{
				cuExecute(num,
					SIUFS_CalculateNewPosition,
					mPosNext,
					mYStar,
					mX,
					this->inAttribute()->getData(),
					m_matK,
					m_Source,
					mass
				);

			}
			mY.assign(mPosNext);
		}

		if (this->inAttribute()->size() != num)
		{
			cuExecute(num,
				SIUFS_UpdateVelocity,
				this->inVelocity()->getData(),
				mPosNext,
				mX,
				this->inTimeStep()->getValue());
		}
		else
		{
			cuExecute(num,
				SIUFS_UpdateVelocity,
				this->inVelocity()->getData(),
				mPosNext,
				mX,
				this->inAttribute()->getData(),
				this->inTimeStep()->getValue());
		}

	}

	DEFINE_CLASS(SemiImplicitUnifiedFluidSolver);
}
