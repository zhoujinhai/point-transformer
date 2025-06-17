#include <iostream>
#include <vector>
#include <chrono>
#include <string> 
#include <algorithm> 
#include <fstream>
#include <sstream>

#include "onnxruntime_cxx_api.h"


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


inline void swap_float(float& x, float& y) {
    std::swap(x, y);
}

inline void swap_int(int& x, int& y) {
    std::swap(x, y);
}

void reheap(float* dist, int* idx, int k) {
    int root = 0;
    while (true) {
        int child = root * 2 + 1;
        if (child >= k) break;

        if (child + 1 < k && dist[child + 1] > dist[child])
            child++;

        if (dist[root] >= dist[child])
            break;

        swap_float(dist[root], dist[child]);
        swap_int(idx[root], idx[child]);
        root = child;
    }
}

void heap_sort(float* dist, int* idx, int k) {
    for (int i = k - 1; i > 0; i--) {
        swap_float(dist[0], dist[i]);
        swap_int(idx[0], idx[i]);
        reheap(dist, idx, i);
    }
}

void knnquery_cpu_impl(
    int b,
    int nsample,
    const float* xyz,
    const float* new_xyz,
    const int* offset,
    const int* new_offset,
    int* idx
    float* dist2
) {
    for (int bid = 0; bid < b; ++bid) {
        // 计算当前批次的点云范围
        int start_n = (bid == 0) ? 0 : offset[bid - 1];
        int end_n = offset[bid];

        // 计算查询点范围
        int start_m = (bid == 0) ? 0 : new_offset[bid - 1];
        int end_m = new_offset[bid];

        // 处理当前批次的所有查询点
        for (int pt_idx = start_m; pt_idx < end_m; ++pt_idx) {
            // 当前查询点坐标
            float new_x = new_xyz[pt_idx * 3];
            float new_y = new_xyz[pt_idx * 3 + 1];
            float new_z = new_xyz[pt_idx * 3 + 2];

            // 初始化最大堆
            std::vector<float> best_dist(nsample, 1e10f);
            std::vector<int> best_idx(nsample, start_n);

            // 遍历点云块内的点
            for (int i = start_n; i < end_n; ++i) {
                float x = xyz[i * 3];
                float y = xyz[i * 3 + 1];
                float z = xyz[i * 3 + 2];

                // 计算平方距离
                float dx = new_x - x;
                float dy = new_y - y;
                float dz = new_z - z;
                float d2 = dx * dx + dy * dy + dz * dz;
 
                // 更新堆
                if (d2 < best_dist[0]) {
                    best_dist[0] = d2;
                    best_idx[0] = i;
                    reheap(best_dist.data(), best_idx.data(), nsample);
                }
            }

            // 堆排序
            heap_sort(best_dist.data(), best_idx.data(), nsample);

            // 存储结果
            for (int i = 0; i < nsample; ++i) {
                idx[pt_idx * nsample + i] = best_idx[i];
                if (dist2) {
                    dist2[pt_idx * nsample + i] = best_dist[i];
                }
            }
        }
    }
}



struct KNNQueryKernel {
    KNNQueryKernel(const OrtApi& ort_api, const OrtKernelInfo* /*info*/) : ort_(ort_api) {
	}

	void Compute(OrtKernelContext* context);

private:
    const OrtApi& ort_;
};

struct KNNQueryKernelOptionalOnnx : Ort::CustomOpBase<KNNQueryKernelOptionalOnnx, KNNQueryKernel> {
	explicit KNNQueryKernelOptionalOnnx(const char* provider) : provider_(provider) {}

	void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
		return new KNNQueryKernel(api, info);
	};
	const char* GetName() const { return "KNNQuery"; };

	const char* GetExecutionProviderType() const { return provider_; };

	size_t GetInputTypeCount() const { return 5; };
	ONNXTensorElementDataType GetInputType(size_t index/*index*/) const {
		if (index == 0) {
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
		}
		else if (index == 1) { 
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		}
        else if (index == 2) { 
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        else if (index == 3) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
        else if (index == 4) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        }
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	};
	OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
		/*if (index == 1)
			return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;*/
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}

	size_t GetOutputTypeCount() const { return 2; };
	ONNXTensorElementDataType GetOutputType(size_t index) const {
        if (index == 1) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	};
	OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}

private:
	const char* provider_;
};

void KNNQueryKernel::Compute(OrtKernelContext * context) {
    // 获取输入输出张量
    Ort::KernelContext ctx(context);

    // 输入1: 采样点数
    auto nsample_input = ctx.GetInput(0);
    const int nsample = *nsample_input.GetTensorData<int>(); 

    // 输入2: 点云数据 [total_points, 3]
    auto xyz_input = ctx.GetInput(1);
    const float* xyz_data = xyz_input.GetTensorData<float>();
    auto xyz_shape = xyz_input.GetTensorTypeAndShapeInfo().GetShape();
    int64_t total_points = xyz_shape[0];

    // 输入3: 点云数据
    auto new_xyz_input = ctx.GetInput(2);
    const float* new_xyz = new_xyz_input.GetTensorData<float>();
    auto new_xyz_shape = new_xyz_input.GetTensorTypeAndShapeInfo().GetShape();

    // 输入3: 批次偏移量 [batch_size]
    auto offset_input = ctx.GetInput(3);
    const int* offset_data = offset_input.GetTensorData<int>();
    auto offset_shape = offset_input.GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch_size = offset_input.GetTensorTypeAndShapeInfo().GetShape()[0];

    // 输入4: 采样偏移量 [batch_size]
    auto new_offset_input = ctx.GetInput(4);
    const int* new_offset_data = new_offset_input.GetTensorData<int>();
    auto new_offset_shape = new_offset_input.GetTensorTypeAndShapeInfo().GetShape();

    // 准备输出张量
    const int64_t M = new_xyz_input.GetTensorTypeAndShapeInfo().GetShape()[0];
    std::vector<int64_t> idx_shape = { M, static_cast<int>(nsample) };  
    Ort::UnownedValue idx_output = ctx.GetOutput(0, idx_shape);  
    int32_t* idx_ptr = idx_output.GetTensorMutableData<int>();

    bool output_dist = ctx.GetOutputCount() > 1;
    float* dist_ptr = nullptr;
    if (output_dist) {
        Ort::UnownedValue dist_output = ctx.GetOutput(1, idx_shape);
        dist_ptr = dist_output.GetTensorMutableData<float>();
    }

    // 调用核心算法实现
    knnquery_cpu_impl(
        batch_size,
        nsample,
        xyz_data,
        new_xyz,
        offset_data,
        new_offset_data,
        idx_ptr,
	output_dist ? dist_ptr : nullptr  
    );
}


#if defined(_WIN32)
static const wchar_t* MODEL_URI = L"F:/code/onnxruntime/test_fps/QueryAndGroup.onnx";
#else
static const char* MODEL_URI = "F:/code/onnxruntime/test_fps/QueryAndGroup.onnx";
#endif

int main(int argc, char* argv[])
{
    const std::string filename = "F:/code/onnxruntime/test_fps/feats.txt";

    std::vector<std::vector<float>> points;
    ReadData(filename, points);

    Ort::CustomOpDomain custom_op_domain("ai.onnx.contrib");
    KNNQueryKernelOptionalOnnx knnQuery_op("CPUExecutionProvider");
    custom_op_domain.Add(&knnQuery_op);

    Ort::SessionOptions session_options;
    session_options.Add(custom_op_domain);
    Ort::Env env_ = Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");
    Ort::Session session(env_, MODEL_URI, session_options);

    std::vector<Ort::Value> ort_outputs;
    const char* output_name = "outputs";
    std::vector<const char*> input_names = { "feats" };

    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    int numPts = points.size();
    int channel = points[0].size();
    std::vector<float> input0data(numPts * channel);
    std::vector<int64_t> input0dims = { numPts, channel };
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
    try {
        ort_outputs = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
    }
    catch (const Ort::Exception& e) {
        // 捕获ONNX Runtime特定的异常
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        return -1;
    }
    Ort::Value& output_tensor = ort_outputs.at(0);
    Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto out_shape = output_info.GetShape();
    size_t total_len = output_info.GetElementCount();
    float* output_data = output_tensor.GetTensorMutableData<float>();
    std::vector<float> idx;
    for (int i = 0; i < total_len; ++i) {
        idx.push_back(output_data[i]);
    }
    return 0;
}
