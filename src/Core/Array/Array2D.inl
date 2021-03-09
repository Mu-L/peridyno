namespace dyno {

	template<typename T>
	void Array2D<T, DeviceType::GPU>::resize(size_t nx, size_t ny)
	{
		if (nullptr != m_data) clear();

		cuSafeCall(cudaMallocPitch((void**)&m_data, &m_pitch, sizeof(T) * nx, ny));
		
		m_nx = nx;	
		m_ny = ny;
		m_pitch /= sizeof(T);
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::reset()
	{
		cuSafeCall(cudaMemset((void*)m_data, 0, m_pitch * m_ny * sizeof(T)));
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::clear()
	{
		if (m_data != nullptr)
			cuSafeCall(cudaFree((void*)m_data));

		m_nx = 0;
		m_ny = 0;
		m_pitch = 0;
		m_data = nullptr;
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::assign(const Array2D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()){
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch, src.data(), src.pitch(), src.nx(), src.ny(), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::assign(const Array2D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch, src.data(), src.nx(), src.nx(), src.ny(), cudaMemcpyHostToDevice));
	}


	template<typename T>
	void Array2D<T, DeviceType::CPU>::resize(size_t nx, size_t ny)
	{
		if (m_data.size() != 0) clear();
		
		m_data.resize(nx*ny);
		m_nx = nx;
		m_ny = ny;
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::reset()
	{
		std::fill(m_data.begin(), m_data.end(), 0);
	}

	template<typename T>
	void dyno::Array2D<T, DeviceType::CPU>::clear()
	{
		m_data.clear();

		m_nx = 0;
		m_ny = 0;
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::assign(const Array2D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), m_nx, src.data(), src.pitch(), src.nx(), src.ny(), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::assign(const Array2D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), m_nx, src.data(), m_nx, src.nx(), src.ny(), cudaMemcpyHostToHost));
	}
}