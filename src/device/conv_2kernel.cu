
// input size: (height_in * width_in * channel_in)
// data size: (hw_out * hw_kernel * channel_in)

__global__ void im2col (float* input, float* data, int height_in, int width_in, int channel_in, int height_kernel, int width_kernel, 
			int height_out, int width_out, int channel_out, int stride)
{	
	int i = blockIdx.y * blockDim.y + threadIdx.y;   // row: 0 - hw_out
	int j = blockIdx.x * blockDim.x + threadIdx.x;   // col: 0 - channel_out
	
	int hw_in = height_in * width_in;
	int hw_kernel = height_kernel * width_kernel;
	int hw_out = height_out * width_out;
	
	if (i < hw_out && j < channel_out)
	{
		if (threadIdx.x == 0)
		{
			for (int c = 0; c < channel_in; c++) 
			{
				int step_h = i / width_out;
				int step_w = i % width_out;
				int start_idx = step_h * width_in * stride + step_w * stride;  
				for (int k = 0; k < hw_kernel; k ++) 
				{
					int cur_col = start_idx % width_in + k % width_kernel; 
					int cur_row = start_idx / width_in + k / width_kernel;
					if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) 
					{
						data[i * hw_kernel * channel_in + c * hw_kernel + k] = 0;
					}
					else 
					{
						int pick_idx = hw_in * c + cur_row * width_in + cur_col;
						data[i * hw_kernel * channel_in + c * hw_kernel + k] = input[pick_idx];
					}
				}	
			}
		}
	}
}

// data size (m, n) - (hw_out, hw_kernel * channel_in)
// weight size (n, k) - (hw_kernel * channel_in, channel_out)
// output size (m, k) - (hw_out, channel_out)
// bias size (k) - (channel_out)

__global__ void convolution (float* data, float* weight, float* output, float* bias, int m, int n, int k)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < k)
	{
		float s = 0;
		for (int p = 0; p < n; p++)
		{
			s += data[i * n + p] * weight[p * k + j];
		}
		output[i * k + j] = s + bias(j);
	        // output[i * k + j] = s;
	}
}

__global__ void convolution_kernel2 (float* data, float* weight, float* output, float* bias, int m, int n, int k)
{
	__shared__ float s_data[TILE_WIDTH][TILE_WIDTH];    //BLOCK HEIGHT, BLOCK WIDTH
	__shared__ float s_weight[TILE_WIDTH][TILE_WIDTH];  //BLOCK HEIGHT, BLOCK WIDTH
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float s = 0;
	
	for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; t++)
	{
		if (i < m && t * TILE_WIDTH + tx < n)
		{
			s_data[ty][tx] = data[i * n + t * TILE_WIDTH + tx];
		}
		else
		{
			s_data[ty][tx] = 0;
		}
		
		if (t * TILE_WIDTH + ty < n && j < k)
		{
			s_weight[ty][tx] = weight[(t * TILE_WIDTH + ty) * k + j];
		}
		else
		{
			s_weight[ty][tx] = 0;
		}
		__syncthreads();
		
		
		for (int p = 0; p < TILE_WIDTH; p++)
		{
			s+= s_data[ty][p] * s_weight[p][tx];
		}
		__syncthreads();
	}	
		
	if (i < m && j < k)
	{
		output[i * k + j] = s + bias(j);
		// output[i * k + j] = s;
	}
}
void Conv::forward(const Matrix& bottom, bool useDevice = false) 
{
	int n_sample = bottom.cols();
	top.resize(height_out * width_out * channel_out, n_sample);
	if (useDevice == false)
	{
		data_cols.resize(n_sample);
		for (int i = 0; i < n_sample; i ++) 
		{
			// im2col
			Matrix data_col;
			im2col(bottom.col(i), data_col);
			data_cols[i] = data_col;

			// conv by product
			Matrix result = data_col * weight;  // result: (hw_out, channel_out)
			result.rowwise() += bias.transpose();
			//Matrix result = matrixMul(data_col, weight);
			//matrixRowwiseAddVec(result, bias);

			top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
		}
	}
	else
	{
		for (int n = 0; n < n_sample; n++) 
		{
			Vector in = bottom.col(n);
			//TODO: Copy data from in to d_input
			
			//TODO: Copy data from weight to d_weight
			
			//Allocate memories
			float* d_data, * d_output;
			size_t n_data = height_out * width_out * height_kernel * width_kernel * channel_in * sizeof(float);
			size_t n_output = height_out * width_out * channel_out * sizeof(float);
			CHECK(cudaMalloc(&d_data, n_data));
			CHECK(cudaMalloc(&d_output, n_output));
			
			//Grid size and Block size
			dim3 blockSize (32, 32); //default
			dim3 gridSize((channel_out - 1) / blockSize.x + 1,
				      (height_out * width_out - 1) / blockSize.y + 1);
			
			im2col<<<gridSize, blockSize>>>(d_input, d_data, height_in, width_in, height_kernel, width_kernel, stride);
			convolution<<<gridSize, blockSize>>>(d_data, d_weight, d_output, d_bias, height_out * width_out, 
							     height_kernel * width_kernel * channel_in, channel_out);
			
			//TODO: Copy data from d_output to out
			Vector out;
			out.resize(height_out * width_out * channel_out);
			
			//TODO: Add out to top.col(n)
		}
	}
}
