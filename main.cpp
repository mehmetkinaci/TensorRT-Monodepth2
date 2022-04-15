#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include "common.hpp"
#include "cuda_utils.h"


#include <opencv2/opencv.hpp>

#define BATCH_SIZE 1

// giriş çıkış boyutları
static const int INPUT_H = 192;
static const int INPUT_W = 640;
static const int OUTPUT_SIZE = 640*192;


const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
static Logger gLogger;


std::map<std::string, Weights> destiny_load_weights(const std::string file)
{
	std::map<std::string, Weights> weightMap;
	std::ifstream input(file);
    assert(input.is_open() && "Weight dosyasini yüklerken bir hata olustu..");
	int32_t count;
    input >> count;
    assert(count > 0 && "Weight dosyasi bozuk, göz at...");
	std::cout<<"Weight dosyasi cout:"<<count<<std::endl;
	while(count--)
	{
		Weights wt{DataType::kFLOAT, nullptr, 0};
		uint32_t size;
		std::string name;
        input >> name >> std::dec >> size;
		wt.type = DataType::kFLOAT;
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;

}

//formul: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
IScaleLayer* destiny_bn2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
	float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;
	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1 && "Batch Norm Layer'inda scale hatasi");
    return scale_1;
}

IActivationLayer* destiny_resnet_make_layer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1 && "conv1 hatasi");
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = destiny_bn2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1 && "relu1 hatasi");
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn2 = destiny_bn2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);
    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = destiny_bn2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    //relu2=output
    IActivationLayer* output = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(output && "relu2 hatasi");
    return output;
}



ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Boyutları { 3, INPUT_H, INPUT_W } ve ismi INPUT_BLOB_NAME olan input tensorü oluşturur. 
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = destiny_load_weights("../encoder.wts");
    std::map<std::string, Weights> weightMap2 = destiny_load_weights("../decoder.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});
    IScaleLayer* bn1 = destiny_bn2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* relu2 = destiny_resnet_make_layer(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    IActivationLayer* relu3 = destiny_resnet_make_layer(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

    IActivationLayer* relu4 = destiny_resnet_make_layer(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    IActivationLayer* relu5 = destiny_resnet_make_layer(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

    IActivationLayer* relu6 = destiny_resnet_make_layer(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    IActivationLayer* relu7 = destiny_resnet_make_layer(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

    IActivationLayer* relu8 = destiny_resnet_make_layer(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    IActivationLayer* relu9 = destiny_resnet_make_layer(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    //////layer4
    auto decoder_pad1 = addDestinyPadLayer(network, 512,6,20,std::vector<IActivationLayer*>{relu9});
    IConvolutionLayer* decoder_conv1 = network->addConvolutionNd(*decoder_pad1->getOutput(0), 256, DimsHW{3, 3}, weightMap2["decoder.0.conv.conv.weight"], emptywts);
    decoder_conv1->setBiasWeights(weightMap2["decoder.0.conv.conv.bias"]);
    IActivationLayer* decoder_elu1 = network->addActivation(*decoder_conv1->getOutput(0), ActivationType::kELU);
    auto decoder_upsample1 = network->addResize(*decoder_elu1->getOutput(0));
    decoder_upsample1->setResizeMode(ResizeMode::kNEAREST);
    decoder_upsample1->setOutputDimensions(Dims3{256,12,40});
    ITensor* decoder_list1[] = { decoder_upsample1->getOutput(0), relu7->getOutput(0) };
    auto decoder_cat1 = network->addConcatenation(decoder_list1, 2);
    auto decoder_pad2 = addDestinyPadLayer(network, 512,12,40,std::vector<IConcatenationLayer*>{decoder_cat1});
    IConvolutionLayer* decoder_conv2 = network->addConvolutionNd(*decoder_pad2->getOutput(0), 256, DimsHW{3, 3}, weightMap2["decoder.1.conv.conv.weight"], emptywts);
    decoder_conv2->setBiasWeights(weightMap2["decoder.1.conv.conv.bias"]);
    IActivationLayer* decoder_elu2 = network->addActivation(*decoder_conv2->getOutput(0), ActivationType::kELU);
    //////layer3
    auto decoder_pad3 = addDestinyPadLayer(network, 256,12,40,std::vector<IActivationLayer*>{decoder_elu2});
    IConvolutionLayer* decoder_conv3 = network->addConvolutionNd(*decoder_pad3->getOutput(0), 128, DimsHW{3, 3}, weightMap2["decoder.2.conv.conv.weight"], emptywts);
    decoder_conv3->setBiasWeights(weightMap2["decoder.2.conv.conv.bias"]);
    IActivationLayer* decoder_elu3 = network->addActivation(*decoder_conv3->getOutput(0), ActivationType::kELU);
    auto decoder_upsample2 = network->addResize(*decoder_elu3->getOutput(0));
    decoder_upsample2->setResizeMode(ResizeMode::kNEAREST);
    decoder_upsample2->setOutputDimensions(Dims3{128,24,80});
    ITensor* decoder_list2[] = { decoder_upsample2->getOutput(0), relu5->getOutput(0) };
    auto decoder_cat2 = network->addConcatenation(decoder_list2, 2);
    auto decoder_pad4 = addDestinyPadLayer(network, 256,24,80,std::vector<IConcatenationLayer*>{decoder_cat2});
    IConvolutionLayer* decoder_conv4 = network->addConvolutionNd(*decoder_pad4->getOutput(0), 128, DimsHW{3, 3}, weightMap2["decoder.3.conv.conv.weight"], emptywts);
    decoder_conv4->setBiasWeights(weightMap2["decoder.3.conv.conv.bias"]);
    IActivationLayer* decoder_elu4 = network->addActivation(*decoder_conv4->getOutput(0), ActivationType::kELU);
    //////layer2
    auto decoder_pad5 = addDestinyPadLayer(network, 128,24,80,std::vector<IActivationLayer*>{decoder_elu4});//VECTOR ICI DUZELT iconcateation layer
    IConvolutionLayer* decoder_conv5 = network->addConvolutionNd(*decoder_pad5->getOutput(0), 64, DimsHW{3, 3}, weightMap2["decoder.4.conv.conv.weight"], emptywts);
    decoder_conv5->setBiasWeights(weightMap2["decoder.4.conv.conv.bias"]);
    IActivationLayer* decoder_elu5 = network->addActivation(*decoder_conv5->getOutput(0), ActivationType::kELU);
    auto decoder_upsample3 = network->addResize(*decoder_elu5->getOutput(0));
    decoder_upsample3->setResizeMode(ResizeMode::kNEAREST);
    decoder_upsample3->setOutputDimensions(Dims3{64,48,160});
    ITensor* decoder_list3[] = { decoder_upsample3->getOutput(0), relu3->getOutput(0) };
    auto decoder_cat3 = network->addConcatenation(decoder_list3, 2);
    auto decoder_pad6 = addDestinyPadLayer(network, 128,48,160,std::vector<IConcatenationLayer*>{decoder_cat3});
    IConvolutionLayer* decoder_conv6 = network->addConvolutionNd(*decoder_pad6->getOutput(0), 64, DimsHW{3, 3}, weightMap2["decoder.5.conv.conv.weight"], emptywts);
    decoder_conv6->setBiasWeights(weightMap2["decoder.5.conv.conv.bias"]);
    IActivationLayer* decoder_elu6 = network->addActivation(*decoder_conv6->getOutput(0), ActivationType::kELU);
    //////layer1
    auto decoder_pad7 = addDestinyPadLayer(network, 64,48,160,std::vector<IActivationLayer*>{decoder_elu6});//VECTOR ICI DUZELT iconcateation layer
    IConvolutionLayer* decoder_conv7 = network->addConvolutionNd(*decoder_pad7->getOutput(0), 32, DimsHW{3, 3}, weightMap2["decoder.6.conv.conv.weight"], emptywts);
    decoder_conv7->setBiasWeights(weightMap2["decoder.6.conv.conv.bias"]);
    IActivationLayer* decoder_elu7 = network->addActivation(*decoder_conv7->getOutput(0), ActivationType::kELU);
    auto decoder_upsample4 = network->addResize(*decoder_elu7->getOutput(0));
    decoder_upsample4->setResizeMode(ResizeMode::kNEAREST);
    decoder_upsample4->setOutputDimensions(Dims3{32,96,320});
    ITensor* decoder_list4[] = { decoder_upsample4->getOutput(0), relu1->getOutput(0) };
    auto decoder_cat4 = network->addConcatenation(decoder_list4, 2);
    auto decoder_pad8 = addDestinyPadLayer(network, 96,96,320,std::vector<IConcatenationLayer*>{decoder_cat4});
    IConvolutionLayer* decoder_conv8 = network->addConvolutionNd(*decoder_pad8->getOutput(0), 32, DimsHW{3, 3}, weightMap2["decoder.7.conv.conv.weight"], emptywts);
    decoder_conv8->setBiasWeights(weightMap2["decoder.7.conv.conv.bias"]);
    IActivationLayer* decoder_elu8 = network->addActivation(*decoder_conv8->getOutput(0), ActivationType::kELU);
    //////layer0
    auto decoder_pad9 = addDestinyPadLayer(network, 32,96,320,std::vector<IActivationLayer*>{decoder_elu8});//VECTOR ICI DUZELT iconcateation layer
    IConvolutionLayer* decoder_conv9 = network->addConvolutionNd(*decoder_pad9->getOutput(0), 16, DimsHW{3, 3}, weightMap2["decoder.8.conv.conv.weight"], emptywts);
    decoder_conv9->setBiasWeights(weightMap2["decoder.8.conv.conv.bias"]);
    IActivationLayer* decoder_elu9 = network->addActivation(*decoder_conv9->getOutput(0), ActivationType::kELU);
    auto decoder_upsample5 = network->addResize(*decoder_elu9->getOutput(0));
    decoder_upsample5->setResizeMode(ResizeMode::kNEAREST);
    decoder_upsample5->setOutputDimensions(Dims3{16,192,640});
    auto decoder_pad10 = addDestinyPadLayer(network, 16,192,640,std::vector<IResizeLayer*>{decoder_upsample5});
    IConvolutionLayer* decoder_conv10 = network->addConvolutionNd(*decoder_pad10->getOutput(0), 16, DimsHW{3, 3}, weightMap2["decoder.9.conv.conv.weight"], emptywts);
    decoder_conv10->setBiasWeights(weightMap2["decoder.9.conv.conv.bias"]);
    IActivationLayer* decoder_elu10 = network->addActivation(*decoder_conv10->getOutput(0), ActivationType::kELU);


    auto decoder_pad11 = addDestinyPadLayer(network, 16,192,640,std::vector<IActivationLayer*>{decoder_elu10});
    IConvolutionLayer* decoder_conv11 = network->addConvolutionNd(*decoder_pad11->getOutput(0), 1, DimsHW{3, 3}, weightMap2["decoder.10.conv.weight"], emptywts);
    decoder_conv11->setBiasWeights(weightMap2["decoder.10.conv.bias"]);
    IActivationLayer* decoder_sigmoid = network->addActivation(*decoder_conv11->getOutput(0), ActivationType::kSIGMOID);

    decoder_sigmoid->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decoder_sigmoid->getOutput(0));
    std::cout << "set name out" << std::endl;

    // engine build edilir
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(BuilderFlag::kFP16);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // network imha edilir
    network->destroy();

    // bellek boşaltılır
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // builder oluşturulur
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // outputları ayarlanıp modele uygun engine build edilir.
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // engine'i serialize eder
    (*modelStream) = engine->serialize();

    // imha edilir
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    //input output sayısı kadar pointer tanımlanır
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // input ve output bufferları ismine göre bind'lanmaktadır
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // GPU bufferları oluşturulur
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // stream oluşturulur
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // stream ve bufferlar boşaltılır
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(){
	cv::Mat image;

    std::string engine_name = "encoder";

    //buradaki kod her mimari değişikliğinde bir kere yorum dışına alıp çalıştırılmalıdır. Bu şekilde model mimarisi serialize edilir
    /*if (1) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }*/
    // Deserialize bu kısımda edilir. Ardından kod görseller üzerinde çalıştırılır.
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

	image=cv::imread("/home/mehmet/destiny/experimental/frame_640x192.jpg",cv::IMREAD_COLOR)/*.convertTo(image,CV_32FC3,(1./255.))*/;

	std::cout<<"Rows: "<<image.rows <<"  Cols:"<<image.cols<<" Dims:"<<image.dims<<"	Channels: "<<image.channels()<<std::endl;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

	static float data[3 * INPUT_H * INPUT_W];
    static float prob[OUTPUT_SIZE];
	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {//burada yapılan işlemler sonucunda 3 renk layerı uc ucua eklenmiş şekilde 1d array döner.
		uchar* uc_pixel = image.data + row * image.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (((float)uc_pixel[2] / 255.0)-0.45)/0.225;
			data[i + INPUT_H * INPUT_W] = (((float)uc_pixel[1] / 255.0)-0.45)/0.225;
			data[i + 2 * INPUT_H * INPUT_W] = (((float)uc_pixel[0] / 255.0)-0.45)/0.225;
			uc_pixel += 3;
			++i;//her pixele bakar her pixel içinde 3 tane değer vardır.RGB
		}
	}
    for(int i=0;i<20;i++){ //Kod 20 kere çalıştırılarak birikmiş batch üzerinden maksimumu performans ölçülür
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout <<"\n"<< 1000/(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << "fps" << std::endl;
    }
    int n = sizeof(prob) / sizeof(float);
    for(int i=0;i<n;i++)
        prob[i] *=256;
    cv::imwrite("./output.jpg",cv::Mat(192,640,CV_32FC1,prob));
	return 1;
}