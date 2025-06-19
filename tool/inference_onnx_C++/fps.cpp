#include <iostream>
#include <vector>
#include <chrono>
#include <string> 
#include <algorithm> 
#include <fstream>
#include <sstream>

#include "onnxruntime_cxx_api.h"

/* data deal
core::Vector3 mean;
float x = 0, y = 0, z = 0;
for (int i = 0; i < numPts; i++) {
	x += points[i][0];
	y += points[i][1];
	z += points[i][2];
}
x /= float(numPts);
y /= float(numPts);
z /= float(numPts);
float maxSqrt = std::numeric_limits<float>::min();
for (int i = 0; i < numPts; i++) {
	points[i][0] -= x;
	points[i][1] -= y;
	points[i][2] -= z;
	float sum = std::sqrt(points[i][0] * points[i][0] + points[i][1] * points[i][1] + points[i][2] * points[i][2]);;
	if (sum > maxSqrt)
		maxSqrt = sum;
}

float x_min = std::numeric_limits<float>::lowest();
float y_min = std::numeric_limits<float>::lowest();
float z_min = std::numeric_limits<float>::lowest();
for (size_t i = 0; i < numPts; ++i) {
	points[i][0] /= maxSqrt;
	points[i][1] /= maxSqrt;
	points[i][2] /= maxSqrt;
	if (points[i][0] < x_min) {
		x_min = points[i][0];
	}
	if (points[i][0] < x_min) {
		y_min = points[i][1];
	}
	if (points[i][0] < x_min) {
		z_min = points[i][2];
	}
}
for (size_t i = 0; i < numPts; ++i) {
	points[i][0] -= x_min;
	points[i][1] -= y_min;
	points[i][2] -= z_min;
	for (size_t c = 0; c < channel; c++) {
		input0data[channel * i + c] = static_cast<float>(points[i][c]); // 填充数据
	}
}

*/
bool ReadData(const std::string filename, std::vector<std::vector<float>>& data, const int COLS_PER_ROW = 12)
{ 
    // 1. 打开文件并检查状态
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return EXIT_FAILURE;
    }

    // 2. 创建动态容器（行数未知） 
    std::string line;

    // 3. 逐行读取并解析
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;

        // 创建当前行容器（预分配列空间）
        std::vector<float> row;
        row.reserve(COLS_PER_ROW);

        // 使用字符串流解析
        std::istringstream iss(line);
        float value;
        int colCount = 0;

        // 读取固定列数
        while (iss >> value && colCount < COLS_PER_ROW) {
            row.push_back(value);
            colCount++;
        }

        // 校验列数量
        if (colCount != COLS_PER_ROW) {
            std::cerr << "错误：第 " << (data.size() + 1)
                << " 行包含 " << colCount
                << " 列数据（应为 " << COLS_PER_ROW << " 列）" << std::endl;
            continue; // 跳过无效行（或改为return EXIT_FAILURE终止）
        }

        // 移动语义添加行
        data.push_back(std::move(row));
    }

    // 4. 关闭文件
    file.close();

    return true;
}
 
// CPU版本最远点采样核心实现
void furthestsampling_cpu_impl(
    int b,
    const float* xyz,
    const int* offset,
    const int* new_offset,
    float* tmp,
    int* idx
) {
    for (int bid = 0; bid < b; ++bid) {
        // 计算当前批次的点云范围
        int start_n = (bid == 0) ? 0 : offset[bid - 1];
        int end_n = offset[bid];
        int n_points = end_n - start_n;

        // 计算采样范围
        int start_m = (bid == 0) ? 0 : new_offset[bid - 1];
        int end_m = new_offset[bid];
        int n_samples = end_m - start_m;

        // 初始化距离数组
        for (int i = start_n; i < end_n; ++i) {
            tmp[i] = 1e10; // 初始化为大数
        }

        // 第一个采样点选择起始点
        idx[start_m] = start_n;
        int current_idx = start_n;

        // 逐步选择最远点
        for (int j = 1; j < n_samples; ++j) {
            float max_dist = -1;
            int best_idx = start_n;

            // 计算当前点到所有点的距离
            float x1 = xyz[current_idx * 3];
            float y1 = xyz[current_idx * 3 + 1];
            float z1 = xyz[current_idx * 3 + 2];

            // 并行优化：使用OpenMP加速距离计算
#pragma omp parallel for reduction(max:max_dist)
            for (int k = start_n; k < end_n; ++k) {
                float x2 = xyz[k * 3];
                float y2 = xyz[k * 3 + 1];
                float z2 = xyz[k * 3 + 2];

                float dist = (x2 - x1) * (x2 - x1) +
                    (y2 - y1) * (y2 - y1) +
                    (z2 - z1) * (z2 - z1);

                // 更新最小距离
                if (dist < tmp[k]) tmp[k] = dist;

                // 寻找最大最小距离的点
                if (tmp[k] > max_dist) {
                    max_dist = tmp[k];
                    best_idx = k;
                }
            }

            // 记录最远点索引
            current_idx = best_idx;
            idx[start_m + j] = best_idx;
        }
    }
}

struct FurthestSamplingKernel {
    FurthestSamplingKernel(const OrtApi& ort_api, const OrtKernelInfo* /*info*/) : ort_(ort_api) {
	}

	void Compute(OrtKernelContext* context);

    //// onnxruntime >= 1.16.0
    //void ComputeV2(OrtKernelContext* context) {
    //    Compute(context);
    //}

private:
    const OrtApi& ort_;
};

struct FurthestSamplingOptionalOnnx : Ort::CustomOpBase<FurthestSamplingOptionalOnnx, FurthestSamplingKernel> {
	explicit FurthestSamplingOptionalOnnx(const char* provider) : provider_(provider) {}

	void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
		return new FurthestSamplingKernel(api, info);
	};
	const char* GetName() const { return "FurthestSampling"; };

	const char* GetExecutionProviderType() const { return provider_; };

	size_t GetInputTypeCount() const { return 3; };
	ONNXTensorElementDataType GetInputType(size_t index/*index*/) const {
		if (index == 0) {
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		}
		else if (index == 1) { 
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
		}
        else if (index == 2) { 
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	};
	OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
		/*if (index == 1)
			return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;*/
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}

	size_t GetOutputTypeCount() const { return 1; };
	ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	};
	OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}

private:
	const char* provider_;
};

void FurthestSamplingKernel::Compute(OrtKernelContext * context) {
    // 获取输入输出张量
    Ort::KernelContext ctx(context);

    // 输入1: 点云数据 [total_points, 3]
    auto xyz_input = ctx.GetInput(0);
    const float* xyz_data = xyz_input.GetTensorData<float>();
    auto xyz_shape = xyz_input.GetTensorTypeAndShapeInfo().GetShape();
    int64_t total_points = xyz_shape[0];

    // 输入2: 批次偏移量 [batch_size]
    auto offset_input = ctx.GetInput(1);
    const int* offset_data = offset_input.GetTensorData<int>();
    int64_t batch_size = offset_input.GetTensorTypeAndShapeInfo().GetShape()[0];

    // 输入3: 采样偏移量 [batch_size]
    auto new_offset_input = ctx.GetInput(2);
    const int* new_offset_data = new_offset_input.GetTensorData<int>();

    // 输出: 采样点索引 [total_samples]
    auto output = ctx.GetOutput(0, { new_offset_data[batch_size - 1] });
    int* idx_data = output.GetTensorMutableData<int>();

    // 分配临时内存
    std::vector<float> tmp_dist(total_points);
    float* tmp_data = tmp_dist.data();

    // 调用核心算法实现
    furthestsampling_cpu_impl(
        batch_size,
        xyz_data,
        offset_data,
        new_offset_data,
        tmp_data,
        idx_data
    );
}

#if defined(_WIN32)
static const wchar_t* MODEL_URI = L"F:/code/onnxruntime/test_fps/fps.onnx";
#else
static const char* MODEL_URI = "F:/code/onnxruntime/test_fps/fps.onnx";
#endif
int main(int argc, char* argv[])
{
    const std::string filename = "F:/code/onnxruntime/test_fps/feats.txt";

    std::vector<std::vector<float>> points;
    ReadData(filename, points);

    Ort::CustomOpDomain custom_op_domain("ai.onnx.contrib");
    FurthestSamplingOptionalOnnx fps_op("CPUExecutionProvider");
    custom_op_domain.Add(&fps_op);
     
    Ort::SessionOptions session_options;
    session_options.Add(custom_op_domain);
    Ort::Env env_ = Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");
    Ort::Session session(env_, MODEL_URI, session_options);

    std::vector<Ort::Value> ort_outputs;
    const char* output_name = "index";
    std::vector<const char*> input_names = { "feats" };

    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    int numPts = points.size();
    int channel = points[0].size();
    std::vector<float> input0data(numPts * channel);
    std::vector<int64_t> input0dims = {numPts, channel };
    for (size_t i = 0; i < numPts; ++i) { 
        for (size_t c = 0; c < channel; c++) {
            input0data[channel * i + c] = static_cast<float>(points[i][c]); // 填充数据
        }
    }
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        info,
        const_cast<float*>(input0data.data()),
        input0data.size(),
        input0dims.data(),
        input0dims.size()
        ));
    ort_outputs = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
    Ort::Value& output_tensor = ort_outputs.at(0);
    Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t total_len = output_info.GetElementCount();
    int* output_data = output_tensor.GetTensorMutableData<int>();
    std::vector<int> idx;
    for (int i = 0; i < total_len; ++i) {
        idx.push_back(output_data[i]);
    }
	return 0;
}
