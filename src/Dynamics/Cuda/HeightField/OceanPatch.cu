#include "OceanPatch.h"

#include "Topology/HeightField.h"

#include "Mapping/HeightFieldToTriangleSet.h"

#include "Algorithm/CudaRand.h"

#include "GLSurfaceVisualModule.h"

#include "Math/Lerp.h"

#include <math_constants.h>

#include <fstream>

namespace dyno
{
    template<typename TDataType>
    OceanPatch<TDataType>::OceanPatch()
        : Node()
    {
        this->setAutoHidden(true);

        auto heights = std::make_shared<HeightField<TDataType>>();
        this->stateHeightField()->setDataPtr(heights);

        std::ifstream input(getAssetPath() + "windparam.txt", std::ios::in);
        for (int i = 0; i <= 12; i++)
        {
            WindParam param;
            int       dummy;
            input >> dummy;
            input >> param.windSpeed;
            input >> param.A;
            input >> param.choppiness;
            input >> param.global;
            mParams.push_back(param);
        }
        mSpectrumWidth = this->varResolution()->getValue() + 1;
        mSpectrumHeight = this->varResolution()->getValue() + 4;

        auto callback = std::make_shared<FCallBackFunc>(std::bind(&OceanPatch<TDataType>::resetWindType, this));

        this->varWindType()->attach(callback);
        this->varWindType()->setRange(0, 12);
        this->varWindType()->setValue(5);

        this->varWindDirection()->setRange(0, 360);
        
		auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
		this->stateHeightField()->connect(mapper->inHeightField());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->varBaseColor()->setValue(Color::Blue1());
		sRender->varUseVertexNormal()->setValue(true);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
    }

    template<typename TDataType>
    OceanPatch<TDataType>::~OceanPatch()
    {
        mH0.clear();
        mHt.clear();
        mDxt.clear();
        mDzt.clear();
    }

	template<typename Real>
	__device__  Complex<Real> complex_exp(Real arg)
	{
		return Complex<Real>(cosf(arg), sinf(arg));
	}

	// generate wave heightfield at time t based on initial heightfield and dispersion relationship
	template <typename Real, typename Complex>
	__global__ void OP_GenerateSpectrumKernel(
		DArray2D<Complex> h0,
		DArray2D<Complex> ht,
		unsigned int    in_width,
		unsigned int    out_width,
		unsigned int    out_height,
		Real           t,
		Real           patchSize)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int in_index = y * in_width + x;
		unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);  // mirrored
		unsigned int out_index = y * out_width + x;

		// calculate wave vector
		Complex k((-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize), (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize));

		// calculate dispersion w(k)
		Real k_len = k.normSquared();
		Real w = sqrtf(9.81f * k_len);

		if ((x < out_width) && (y < out_height))
		{
			Complex h0_k = h0(x, y);
			Complex h0_mk = h0(out_width - x, out_height - y);

			// output frequency-space complex values
			ht[out_index] = h0_k * complex_exp(w * t) + h0_mk.conjugate() * complex_exp(-w * t);
		}
	}

	template <typename Real, typename Complex>
	__global__ void OP_GenerateDispalcementKernel(
		DArray2D<Complex>      ht,
		DArray2D<Complex>      Dxt,
		DArray2D<Complex>      Dzt,
		unsigned int width,
		unsigned int height,
		Real patchSize)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		// calculate wave vector
		Real kx = (-(int)width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
		Real ky = (-(int)height / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);
		Real k_squared = kx * kx + ky * ky;
		if (k_squared == 0.0f)
		{
			k_squared = 1.0f;
		}
		kx = kx / sqrtf(k_squared);
		ky = ky / sqrtf(k_squared);

		Complex ht_ij = ht(x, y);
		Complex idoth = Complex(-ht_ij.imagPart(), ht_ij.realPart());

		Dxt(x, y) = kx * idoth;
		Dzt(x, y) = ky * idoth;
	}

    template<typename TDataType>
    void OceanPatch<TDataType>::resetWindType()
    {
        int windType = this->varWindType()->getValue();
        this->varAmplitude()->setValue(mParams[windType].A);
        this->varWindSpeed()->setValue(mParams[windType].windSpeed);
        this->varChoppiness()->setValue(mParams[windType].choppiness);
        this->varGlobalShift()->setValue(mParams[windType].global);
    }

	template <typename Real, typename Complex>
	__global__ void OP_GenerateH0(
		DArray2D<Complex> h0,
        Real winDir,
        Real winSpeed,
        Real winDirDependence,
        Real amplitude,
        Real patchSize,
        Real G,
        uint res)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= h0.nx() || y >= h0.ny()) return;

		Real scaledWinDir = winDir * (M_PI / Real(180));

        RandNumber rd(x * y);

		auto phillips = [=](Real Kx, Real Ky, Real Vdir, Real V, Real A, Real dir_depend) -> Real
			{
				Real k_squared = Kx * Kx + Ky * Ky;

				if (k_squared == 0.0f)
				{
					return 0.0f;
				}

				// largest possible wave from constant wind of velocity v
				Real L = V * V / G;

				Real k_x = Kx / std::sqrt(k_squared);
				Real k_y = Ky / std::sqrt(k_squared);
				Real w_dot_k = k_x * std::cos(Vdir) + k_y * std::sin(Vdir);

				Real phillips = A * std::exp(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

				// filter out waves moving opposite to wind
				if (w_dot_k < 0.0f)
				{
					phillips *= dir_depend;
				}

				return phillips;
			};

		auto gauss = [&]() -> Real
			{
				Real u1 = rd.Generate();
				Real u2 = rd.Generate();

				if (u1 < EPSILON)
				{
					u1 = EPSILON;
				}

				return glm::sqrt(-2 * glm::log(u1)) * glm::cos(2 * CUDART_PI_F * u2);
			};

		Real kx = (-(int)res / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
		Real ky = (-(int)res / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

        Real P = glm::sqrt(phillips(kx, ky, scaledWinDir, winSpeed, amplitude, winDirDependence));

		if (glm::abs(kx) < EPSILON && glm::abs(ky) == EPSILON)
		{
			P = 0.0f;
		}

		Real Er = gauss();
		Real Ei = gauss();

		Real h0_re = Er * P * CUDART_SQRT_HALF_F;
		Real h0_im = Ei * P * CUDART_SQRT_HALF_F;

		h0(x, y) = Complex(h0_re, h0_im);
	}

    template<typename TDataType>
    void OceanPatch<TDataType>::resetStates()
    {
        uint res = this->varResolution()->getValue();

        cufftPlan2d(&fftPlan, res, res, CUFFT_C2C);

        int spectrumSize = mSpectrumWidth * mSpectrumHeight * sizeof(Complex);
        mH0.resize(mSpectrumWidth, mSpectrumHeight);

// 		Complex* host_h0 = (Complex*)malloc(spectrumSize);
// 		generateH0(host_h0);
// 
// 		cuSafeCall(cudaMemcpy(mH0.begin(), host_h0, spectrumSize, cudaMemcpyHostToDevice));

        cuExecute2D(make_uint2(mSpectrumWidth, mSpectrumHeight),
            OP_GenerateH0,
            mH0,
            this->varWindDirection()->getValue(),
            this->varWindSpeed()->getValue(),
            mDirDepend,
            this->varAmplitude()->getValue(),
            this->varPatchSize()->getValue(),
            g,
            res);

        mHt.resize(res, res);
        mDxt.resize(res, res);
        mDzt.resize(res, res);
        mDisp.resize(res, res);

        auto topo = this->stateHeightField()->getDataPtr();
        Real h = this->varPatchSize()->getValue() / res;
        topo->setExtents(res, res);
        topo->setGridSpacing(h);
        topo->setOrigin(Vec3f(-0.5 * h * topo->width(), 0, -0.5 * h * topo->height()));

        this->update();
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::updateStates()
    {
        Real timeScaled = this->varTimeScale()->getValue() * this->stateElapsedTime()->getValue();

        uint res = this->varResolution()->getValue();

        cuExecute2D(make_uint2(res, res),
            OP_GenerateSpectrumKernel,
            mH0,
            mHt,
            mSpectrumWidth,
            res,
            res,
            timeScaled,
            this->varPatchSize()->getData());

        cuExecute2D(make_uint2(res, res),
            OP_GenerateDispalcementKernel,
            mHt,
            mDxt,
            mDzt,
            res,
            res,
            this->varPatchSize()->getData());

        cufftExecC2C(fftPlan, (float2*)mHt.begin(), (float2*)mHt.begin(), CUFFT_INVERSE);
        cufftExecC2C(fftPlan, (float2*)mDxt.begin(), (float2*)mDxt.begin(), CUFFT_INVERSE);
        cufftExecC2C(fftPlan, (float2*)mDzt.begin(), (float2*)mDzt.begin(), CUFFT_INVERSE);

        cuExecute2D(make_uint2(res, res),
            O_UpdateDisplacement,
            mDisp,
            mHt,
            mDxt,
            mDzt,
            res);
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::postUpdateStates()
    {
        auto choppiness = this->varChoppiness()->getValue();

        auto topo = this->stateHeightField()->getDataPtr();

        auto h = topo->getGridSpacing();
        auto origin = topo->getOrigin();

        auto& shifts = topo->getDisplacement();

        uint2 extent;
        extent.x = shifts.nx();
        extent.y = shifts.ny();
        cuExecute2D(extent,
            CW_UpdateHeightDisp,
            shifts,
            mDisp,
            origin,
            h,
            choppiness);

       // topo->rasterize();
    }

    template<typename Coord, typename Complex>
    __global__ void O_UpdateDisplacement(
        DArray2D<Coord> displacement,
        DArray2D<Complex> Dh,
        DArray2D<Complex> Dx,
        DArray2D<Complex> Dz,
        int patchSize)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < patchSize && j < patchSize)
        {
            Real sign_correction = ((i + j) & 0x01) ? -1.0f : 1.0f;
            Real h_ij = sign_correction * Dh(i, j).realPart();
            Real x_ij = sign_correction * Dx(i, j).realPart();
            Real z_ij = sign_correction * Dz(i, j).realPart();

            displacement(i, j) = Coord(x_ij, h_ij, z_ij);
        }
    }

    template <typename Real, typename Coord>
    __global__ void CW_UpdateHeightDisp(
        DArray2D<Coord> heights,
        DArray2D<Coord> dis,
        Coord origin,
        Real h,
        Real choppiness)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < heights.nx() && j < heights.ny())
        {
            Real x = (i * h - origin.x) / h;
            Real y = (j * h - origin.y) / h;

            Coord Dij = bilinear(dis, x, y, LerpMode::REPEAT);

            Coord v;
            v[0] = choppiness * Dij[0];
            v[1] = Dij[1];
            v[2] = choppiness * Dij[2];

            heights(i, j) = v;
        }
    }

    DEFINE_CLASS(OceanPatch);
}