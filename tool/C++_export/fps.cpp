//#include <torch/extension.h>
#include <torch/script.h>
#include <vector>
#include <cmath>

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
        int start_n = (bid == 0) ? 0 : offset[bid-1];
        int end_n = offset[bid];
        int n_points = end_n - start_n;

        // 计算采样范围
        int start_m = (bid == 0) ? 0 : new_offset[bid-1];
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
                
                float dist = (x2 - x1)*(x2 - x1) + 
                            (y2 - y1)*(y2 - y1) + 
                            (z2 - z1)*(z2 - z1);
                
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

// 张量接口封装
torch::Tensor furthestsampling_cpu(
    torch::Tensor xyz_tensor,
    torch::Tensor offset_tensor,
    torch::Tensor new_offset_tensor
) {
    // 输入验证
    TORCH_CHECK(xyz_tensor.is_contiguous(), "XYZ tensor must be contiguous");
    TORCH_CHECK(offset_tensor.device().is_cpu(), "Offset tensor must be on CPU");
    TORCH_CHECK(xyz_tensor.size(1) == 3, "XYZ tensor must have shape [N, 3]");

    const int b = offset_tensor.size(0);
    const int n = xyz_tensor.size(0);
    
    // 创建临时距离缓存
    auto tmp_tensor = torch::full({n}, 1e10, 
                                torch::dtype(torch::kFloat32).device(torch::kCPU));
    
    // 创建输出索引
    auto idx_tensor = torch::zeros({new_offset_tensor[-1].item<int>()}, 
                                  torch::dtype(torch::kInt32).device(torch::kCPU));

    // 获取数据指针
    const float* xyz = xyz_tensor.data_ptr<float>();
    const int* offset = offset_tensor.data_ptr<int>();
    const int* new_offset = new_offset_tensor.data_ptr<int>();
    float* tmp = tmp_tensor.data_ptr<float>();
    int* idx = idx_tensor.data_ptr<int>();

    // 执行采样算法
    furthestsampling_cpu_impl(b, xyz, offset, new_offset, tmp, idx);
    
    return idx_tensor;
}

// 模块注册
TORCH_LIBRARY(my_ops, m) {
    m.def("FurthestSampling", furthestsampling_cpu);
}
