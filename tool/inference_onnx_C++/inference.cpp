#if defined(_WIN32)
static const wchar_t* MODEL_URI = L"F:/code/onnxruntime/test_fps/support_sample_aug_4_sim.onnx";
#else
static const char* MODEL_URI = "F:/code/onnxruntime/test_fps/support.onnx";
#endif
int main(int argc, char* argv[])
{
    const std::string filename = "F:/code/onnxruntime/test_fps/1e087f16-b0ce-46bc-9b11-7038ffd13013_Y30_point_sample_aug.txt";

    std::vector<std::vector<float>> points;
    ReadData(filename, points);

    Ort::CustomOpDomain custom_op_domain("ai.onnx.contrib");
    FurthestSamplingOptionalOnnx fps_op("CPUExecutionProvider");
    KNNQueryKernelOptionalOnnx knnQuery_op("CPUExecutionProvider");
    custom_op_domain.Add(&fps_op);
    custom_op_domain.Add(&knnQuery_op);

    Ort::SessionOptions session_options;
    session_options.Add(custom_op_domain);
    Ort::Env env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Default"); //Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");

    std::vector<Ort::Value> ort_outputs;
    const char* output_name = "res";
    std::vector<const char*> input_names = { "inputs" };
     Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    int numPts = points.size();
    std::cout << "Number point: " << numPts << std::endl;
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
        // session_options.EnableProfiling(L"./onnxruntime_profile");
        session_options.SetInterOpNumThreads(4);
        session_options.SetIntraOpNumThreads(4);
        auto start = std::chrono::high_resolution_clock::now(); 
        Ort::Session session(env_, MODEL_URI, session_options);
        auto end = std::chrono::high_resolution_clock::now();  
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
        std::cout << "加载耗时: " << duration << "ms" << std::endl;
        end = std::chrono::high_resolution_clock::now();
        ort_outputs = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
        auto end2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count();
        std::cout << "推理耗时: " << duration << "ms" << std::endl;
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
    std::vector<std::vector<float>> classConfs;
    std::vector<std::vector<int>> classIdx; 
    int number_points = out_shape[0];
    int n_cls = out_shape[1];

    classConfs.resize(n_cls);
    classIdx.resize(n_cls);

    for (int n = 0; n < number_points; ++n) {
        if (points[n][5] >= 0) {
            classIdx[0].push_back(n);
            classConfs[0].push_back(1.0);
        }
        else {
            int maxId = 0;
            float maxProb = output_data[n * n_cls];
            for (int c = 1; c < n_cls; ++c) {
                if (output_data[n * n_cls + c] > maxProb) {
                    maxId = c;
                    maxProb = output_data[n * n_cls + c];
                }
            }
            classIdx[maxId].push_back(n);
            classConfs[maxId].push_back(cv::exp(maxProb));
        }
    }

    return 0;
}
