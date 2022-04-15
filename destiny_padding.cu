#include "destiny_padding.h"
#include "stdio.h"
#include "NvInfer.h"

#include <cudnn.h>
#include <vector>
#include <cuda.h>
#include <string>
#include <vector>
#include "cuda_utils.h"

namespace Tn
{
    template<typename T> 
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> 
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}




#define CUDA_NUM_THREADS 512
dim3 GET_BLOCKS(uint n)
{
    uint k = (n - 1) /CUDA_NUM_THREADS + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*CUDA_NUM_THREADS) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}

namespace nvinfer1
{
    DestinyPaddingPlugin::DestinyPaddingPlugin(int in_channels, int height , int width)
    {
        min_channels=in_channels;
        mheight=height;
        mwidth=width;
        mwidth_out=width+2;
        mheight_out=height+2;
    }

    DestinyPaddingPlugin::~DestinyPaddingPlugin()
    {
    }

    // byte streamde plugin oluşturur
    DestinyPaddingPlugin::DestinyPaddingPlugin(const void* buffer, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(buffer), *a = d;
        read(d, min_channels);
        read(d, mheight);
        read(d, mwidth);
        read(d, mwidth_out);
        read(d, mheight_out);

        assert(d == a + length);
    }

    void DestinyPaddingPlugin::serialize(void* buffer) const TRT_NOEXCEPT
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, min_channels);
        write(d, mheight);
        write(d, mwidth);
        write(d, mwidth_out);
        write(d, mheight_out);

        assert(d == a + getSerializationSize());
    }

    size_t DestinyPaddingPlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return 5*sizeof(int);
    }

    int DestinyPaddingPlugin::initialize() TRT_NOEXCEPT
    { 
        return 0;
    }

    Dims DestinyPaddingPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        //output boyutları verilir
        int dim_in_channels = inputs[0].d[0];
        int dim_height_out = inputs[0].d[1]+2;
        int dim_width_out = inputs[0].d[2]+2;

        return Dims3(dim_in_channels, dim_height_out, dim_width_out);
    }

    // plugin namespace'i belirlenir.
    void DestinyPaddingPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* DestinyPaddingPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    // Çıkışta İndexi verilen verinin tipini dönen fonksiyondur
    DataType DestinyPaddingPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    bool DestinyPaddingPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    bool DestinyPaddingPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void DestinyPaddingPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    void DestinyPaddingPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    void DestinyPaddingPlugin::detachFromContext() TRT_NOEXCEPT {}

    const char* DestinyPaddingPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "Destiny_Padding_TRT";
    }

    const char* DestinyPaddingPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void DestinyPaddingPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    // klonlama işlemi yapılır
    IPluginV2IOExt* DestinyPaddingPlugin::clone() const TRT_NOEXCEPT
    {
        DestinyPaddingPlugin *p = new DestinyPaddingPlugin(min_channels,mwidth,mheight);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }
    //Burada TensorRT'de olmayan Symmetric Pad işlemi CUDA kullanılarak metod olarak eklenmiştir.

    __global__ void symmetric_pad(const float *a, float *c,int *channel,int *height,int *width) {
        int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
        int index= (blockId * blockDim.x) + threadIdx.x;

        
        int k=index/((*height+2)*(*width+2));
        int j=(index%((*height+2)*(*width+2)))/(*width+2);
        int i=(index%((*height+2)*(*width+2)))%(*width+2);

        if(k>=0 && k< *channel){
            if(i>=1 && i<= *width && j>=1 && j<= *height && k< *channel){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(*width)*(j-1)+i-1));

            }
            else if(j==0 && i!=0 && i!= *width+1 && i< *width+1 && i>0){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+ *width+i-1));
            }
            else if(j== *height+1 && i!=0 && i!= *width+1 && i< *width+1 && i>0){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(*width)*(*height-2)+i-1));
            }
            else if(i==0 && j==0){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(*width)+1));
            }
            else if(i== *width+1 && j==0){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+2*(*width)-2)); 
            }
            else if(i== *width+1 && j== *height+1){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(*height-1)*(*width)-2));
            }
            else if(i==0 && j== *height+1){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(*width)*(*height-2)+1));
            }
            else if(i==0 && j < *height+1 && j > 0){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(j-1)*(*width)+1));
            }
            else if(i== *width+1 && j < *height+1 && j > 0){
                *(c+(k * ( *width + 2)* ( *height + 2 )+ ( *width + 2) * j +i )) = *(a+(k*(*width)*(*height)+(j)*(*width)-2));
            }
        }
    }
    void DestinyPaddingPlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize)
    {
        int *width_cuda;
        int *height_cuda;
        int *channel_cuda;
        cudaMalloc((void **)&width_cuda, sizeof(int));
        cudaMalloc((void **)&height_cuda, sizeof(int));
        cudaMalloc((void **)&channel_cuda, sizeof(int));
        cudaMemcpy(width_cuda, &mwidth, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(height_cuda, &mheight, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(channel_cuda, &min_channels, sizeof(int), cudaMemcpyHostToDevice);

        symmetric_pad<<<GET_BLOCKS(min_channels*(mwidth+2)*(mheight+2)),512,0,stream>>>(inputs[0],output,channel_cuda,height_cuda,width_cuda);
    }

    int DestinyPaddingPlugin::enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        //GPU ile işlem tetiklenir
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
        return 0;
    };

    PluginFieldCollection DestinyPaddingPluginCreator::mFC{};
    std::vector<PluginField> DestinyPaddingPluginCreator::mPluginAttributes;

    DestinyPaddingPluginCreator::DestinyPaddingPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* DestinyPaddingPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "Destiny_Padding_TRT";
    }

    const char* DestinyPaddingPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    const PluginFieldCollection* DestinyPaddingPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    IPluginV2IOExt* DestinyPaddingPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
    {
        assert(fc->nbFields == 1);
        assert(strcmp(fc->fields[0].name, "netinfo") == 0);

        int *p_netinfo = (int*)(fc->fields[0].data);

        int input_c=p_netinfo[0];
        int input_h=p_netinfo[1];
        int input_w=p_netinfo[2];
        DestinyPaddingPlugin* obj = new DestinyPaddingPlugin(input_c,input_h,input_w);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* DestinyPaddingPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
    {
        // Network imha edilince bu obje de silinecektir
        // call PReluPlugin::destroy()
        DestinyPaddingPlugin* obj = new DestinyPaddingPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}