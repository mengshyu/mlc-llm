/*!
 *  Copyright (c) 2023-2024 by Contributors
 * \file support/debug_utils.h
 * \brief Tools for debug purposes.
 */
#ifndef MLC_LLM_SUPPORT_IMAGE_UTILS_H_
#define MLC_LLM_SUPPORT_IMAGE_UTILS_H_

#include <tvm/runtime/ndarray.h>

#include <string>

namespace mlc {
namespace llm {

/*! \brief Calculate output shape of resize function for vision model. */
void CalculateResizeShape(tvm::runtime::NDArray image_data, std::string model_type, int &target_height, int &target_width);
void CalculatePadSize(tvm::runtime::NDArray image_data, std::string model_type, int &pad_height, int &pad_width);

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_IMAGE_UTILS_H_
