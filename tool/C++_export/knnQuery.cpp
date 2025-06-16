#include <torch/script.h>
#include <vector>
#include <queue>
#include <cmath>

// 辅助函数：堆操作
inline void swap_float(float &x, float &y) {
    std::swap(x, y);
}

inline void swap_int(int &x, int &y) {
    std::swap(x, y);
}

void reheap(float *dist, int *idx, int k) {
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

void heap_sort(float *dist, int *idx, int k) {
    for (int i = k - 1; i > 0; i--) {
        swap_float(dist[0], dist[i]);
        swap_int(idx[0], idx[i]);
        reheap(dist, idx, i);
    }
}

// KNN核心实现
void knnquery_cpu_impl(
    int b, 
    int nsample,
    const float* xyz,
    const float* new_xyz,
    const int* offset,
    const int* new_offset,
    int* idx,
    float* dist2
) {
    for (int bid = 0; bid < b; ++bid) {
        // 计算当前批次的点云范围
        int start_n = (bid == 0) ? 0 : offset[bid-1];
        int end_n = offset[bid];
        
        // 计算查询点范围
        int start_m = (bid == 0) ? 0 : new_offset[bid-1];
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
                dist2[pt_idx * nsample + i] = best_dist[i];
            }
        }
    }
}

// PyTorch接口函数
std::vector<torch::Tensor> knnquery_cpu(
    int64_t nsample,
    torch::Tensor xyz_tensor,
    torch::Tensor new_xyz_tensor,
    torch::Tensor offset_tensor,
    torch::Tensor new_offset_tensor
) {
    // 输入验证
    TORCH_CHECK(xyz_tensor.is_contiguous(), "XYZ tensor must be contiguous");
    TORCH_CHECK(new_xyz_tensor.is_contiguous(), "New XYZ tensor must be contiguous");
    TORCH_CHECK(offset_tensor.device().is_cpu(), "Offset tensor must be on CPU");
    TORCH_CHECK(xyz_tensor.size(1) == 3, "XYZ tensor must have shape [N, 3]");
    
    // 获取批次数量
    int b = offset_tensor.size(0);
    
    // 准备输出张量
    int m = new_xyz_tensor.size(0);
    auto idx_tensor = torch::zeros({m, nsample}, torch::kInt32);
    auto dist2_tensor = torch::zeros({m, nsample}, torch::kFloat32);
    
    // 获取原始数据指针
    const float* xyz = xyz_tensor.data_ptr<float>();
    const float* new_xyz = new_xyz_tensor.data_ptr<float>();
    const int* offset = offset_tensor.data_ptr<int>();
    const int* new_offset = new_offset_tensor.data_ptr<int>();
    int* idx = idx_tensor.data_ptr<int>();
    float* dist2 = dist2_tensor.data_ptr<float>();
    
    // 执行KNN查询
    knnquery_cpu_impl(b, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
    
    // 返回索引和距离
    return {idx_tensor, torch::sqrt(dist2_tensor)};
}

// 模块注册 
TORCH_LIBRARY(my_ops, m) {
    m.def("KNNQuery", knnquery_cpu);
}

