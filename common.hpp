#ifndef DESTINY_COMMON_H_
#define DESTINY_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "destiny_padding.h"

using namespace nvinfer1;

template<typename T> 
IPluginV2Layer* addDestinyPadLayer(INetworkDefinition *network, int common_c, int common_h, int common_w, std::vector<T*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("Destiny_Padding_TRT", "1");
    PluginField plugin_fields[1];
    int netinfo[4] = {common_c, common_h, common_w};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 3;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection pfc;
    pfc.nbFields = 1;
    pfc.fields = plugin_fields;
    IPluginV2 *pluginObj = creator->createPlugin("Destiny_Padding", &pfc);
    assert(pluginObj != nullptr);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto decodelayer = network->addPluginV2(&input_tensors[0], 1, *pluginObj);
    assert(decodelayer);
    return decodelayer;
}


#endif