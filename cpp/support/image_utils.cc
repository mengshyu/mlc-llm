/*!
 *  Copyright (c) 2023-2024 by Contributors
 * \file support/image_utils.cc
 */
#include "image_utils.h"

#include <cmath>

namespace mlc {
namespace llm {

void CalculateResizeShape(tvm::runtime::NDArray image_data, std::string model_type, int &target_height, int &target_width) {
    ICHECK_EQ(image_data->shape[3], 3) << "Image format must be NHWC";
    int height = image_data->shape[1];
    int width = image_data->shape[2];


    if ("phi3_v" == model_type) {
        const int hd_num = 4;
        double ratio = static_cast<double>(width) / height;
        int scale = 1;
        while (scale * std::ceil(scale / ratio) <= hd_num) {
            scale += 1;
        }
        scale -= 1;
        target_width = static_cast<int>(scale * 336);
        target_height = static_cast<int>(target_width / ratio);
    }
}
}  // namespace llm
}  // namespace mlc
