"""
Implementation for Phi architecture.
"""

from tvm import relax, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor, op
from tvm.script import tir as T

from mlc_llm.model.vision import CLIPVisionModel
from mlc_llm.support.config import ConfigBase


# mypy: disable-error-code="attr-defined"
# pylint: disable=invalid-name,missing-docstring
class ImageProjection(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: ConfigBase):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * 4, config.hidden_size, bias=True
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, image_features: Tensor) -> Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)

        hidden_states = op.wrap_nested(
            relax.BlockBuilder()
            .current()
            .match_cast(
                hidden_states._expr,
                relax.TensorStructInfo(
                    [hidden_states.shape[0], hidden_states.shape[1], 3072], hidden_states.dtype
                ),
            ),
            "hidden_states",
        )

        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Phi3ImageEmbedding(Module):
    def __init__(self, config: ConfigBase):
        super().__init__()

        self.img_processor = CLIPVisionModel(config.vision_config)
        self.image_dim_out = config.img_processor["image_dim_out"]

        self.glb_GN = nn.Parameter((1, 1, self.image_dim_out * 4))
        self.sub_GN = nn.Parameter((1, 1, 1, self.image_dim_out * 4))

        self.img_projection = ImageProjection(config)
        self.image_size = config.vision_config.image_size

    # pylint: disable=dangerous-default-value
    def apply_schedule(self, sch, block, bdx=32, tile=[32, 32]):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[tile[0], None])
        yo, yi = sch.split(loop_y, factors=[tile[1], None])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def dyn_repeat(self, input, repeats, axis) -> Tensor:
        def create_dyn_repeat_func(repeats, axis, dtype):
            @T.prim_func
            def dyn_repeat_func(input: T.handle, output: T.handle):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                input_buf = T.match_buffer(input, (n, c, h, w), dtype=dtype)
                out_buf = T.match_buffer(output, (n, c * repeats, h, w), dtype=dtype)
                out_c = c * repeats

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(out_c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h, w):
                            with T.block("dyn_repeat"):
                                T.reads(input_buf[n_idx, c_idx, h_idx, w_idx])
                                T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                out_buf[n_idx, c_idx, h_idx, w_idx] = input_buf[n_idx, c_idx % c, h_idx, w_idx]
            sch = tir.Schedule(dyn_repeat_func)
            self.apply_schedule(sch, sch.get_block("dyn_repeat"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        assert 1 == axis
        n, c, h, w = input.shape
        out = op.tensor_ir_op(
            create_dyn_repeat_func(repeats, axis, input.dtype),
            "dyn_repeat",
            [input],
            [Tensor.placeholder([n, c * repeats, h, w], input.dtype)],
        )
        return out

    def dyn_concate_dim_2(self, input_1, input_2) -> Tensor:
        def create_dyn_concate_func(dtype):
            @T.prim_func
            def dyn_concate_dim_2_func(input_1: T.handle, input_2: T.handle, output: T.handle):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h1, h2, w = T.int64(), T.int64(), T.int64(), T.int64(), T.int64()
                input_1_buf = T.match_buffer(input_1, (n, c, h1, w), dtype=dtype)
                input_2_buf = T.match_buffer(input_2, (n, c, h2, w), dtype=dtype)
                out_buf = T.match_buffer(output, (n, c, h1 + h2, w), dtype = dtype)

                for n_idx in T.thread_binding(n, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h1 + h2, w):
                            with T.block("dyn_concate_dim_2"):
                                T.reads(input_1_buf[n_idx, c_idx, h_idx, w_idx])
                                T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                if h_idx < h1:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = input_1_buf[n_idx, c_idx, h_idx, w_idx]
                                else:
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = input_2_buf[n_idx, c_idx, h_idx, w_idx]


            sch = tir.Schedule(dyn_concate_dim_2_func)
            self.apply_schedule(sch, sch.get_block("dyn_concate_dim_2"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        n1, c1, h1, w1 = input_1.shape
        n2, c2, h2, w2 = input_2.shape
        out = op.tensor_ir_op(
            create_dyn_concate_func(input_1.dtype),
            "dyn_concate_dim_2",
            [input_1, input_2],
            [Tensor.placeholder([n1, c1, h1 + h2, w1], input_1.dtype)],
        )
        return out


    def dyn_concate_dim_1(self, input_1, input_2) -> Tensor:
        def create_dyn_concate_func(dtype):
            @T.prim_func
            def dyn_concate_dim_1_func(input_1: T.handle, input_2: T.handle, output: T.handle):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                c, h1, h2, w = T.int64(), T.int64(), T.int64(), T.int64()
                input_1_buf = T.match_buffer(input_1, (c, h1, w), dtype=dtype)
                input_2_buf = T.match_buffer(input_2, (c, h2, w), dtype=dtype)
                out_buf = T.match_buffer(output, (c, h1 + h2, w), dtype = dtype)

                for n_idx in T.thread_binding(1, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h1+h2, w):
                            with T.block("dyn_concate_dim_1"):
                                T.reads(input_1_buf[c_idx, h_idx, w_idx])
                                T.writes(out_buf[c_idx, h_idx, w_idx])
                                if h_idx < h1:
                                    out_buf[c_idx, h_idx, w_idx] = input_1_buf[c_idx, h_idx, w_idx]
                                else:
                                    out_buf[c_idx, h_idx, w_idx] = input_2_buf[c_idx, h_idx, w_idx]


            sch = tir.Schedule(dyn_concate_dim_1_func)
            self.apply_schedule(sch, sch.get_block("dyn_concate_dim_1"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        c1, h1, w1 = input_1.shape
        c2, h2, w2 = input_2.shape
        out = op.tensor_ir_op(
            create_dyn_concate_func(input_1.dtype),
            "dyn_concate",
            [input_1, input_2],
            [Tensor.placeholder([c1, h1 + h2, w1], input_1.dtype)],
        )
        return out

    def get_img_features(self, img_embeds: Tensor) -> Tensor:
        img_processor_output = self.img_processor(img_embeds)
        patch_feature = nn.op.split(img_processor_output, indices_or_sections=[1], axis=1)
        return patch_feature[1]


    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        N, L, C = image_features.shape
        num_images = 1
        H = int(L**0.5)
        image_features = nn.op.reshape(image_features, ([N, H, H, C]))
        image_features = nn.op.reshape(image_features, ([N, H // 2, 2, H // 2, 2, C]))
        image_features = nn.op.permute_dims(image_features, axes=([0, 1, 3, 2, 4, 5]))
        image_features = nn.op.reshape(image_features, ([num_images, -1, 4 * C]))
        image_features = nn.op.reshape(image_features, ([num_images, h_crop, w_crop, H // 2, H // 2, -1]))
        image_features = nn.op.permute_dims(image_features, axes=([0, 1, 3, 2, 4, 5]))
        image_features_hd = nn.op.reshape(image_features, ([num_images, (h_crop * H) // 2, (w_crop * H) // 2, 4 * C]))
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape

        temp_sub_GN = self.dyn_repeat(self.sub_GN, 1, 1)
        image_features_hd_newline = self.dyn_concate_dim_2(image_features_hd, temp_sub_GN)
        image_features_hd_newline = nn.op.reshape(image_features_hd_newline, ([num_images, -1, hid_dim]))
        return image_features_hd_newline

    # pylint: disable=too-many-locals,too-many-locals,unused-argument
    def forward(self, pixel_values: Tensor, raw_image_h, raw_image_w) -> Tensor:

        img_features = self.get_img_features(pixel_values)
        H = T.int32((img_features.shape[1] ** 0.5))
        img_features = nn.op.split(img_features, indices_or_sections=[1], axis=0)
        global_image_features = img_features[0]
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)

        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        h_crop =  raw_image_h // self.image_size
        w_crop =  raw_image_w // self.image_size
        num_crops = h_crop * w_crop
        sub_image_features = img_features[1]
        sub_image_features_hd = self.reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop)
        sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

        global_image_features_hd = nn.op.squeeze(global_image_features_hd, 0)

        output_img = self.dyn_concate_dim_1(sub_image_features_hd_newline, self.glb_GN)
        output_img = self.dyn_concate_dim_1(output_img, global_image_features_hd)
        img_set_tensor = self.img_projection(output_img)
        img_set_tensor = nn.op.squeeze(img_set_tensor, 0)
        return img_set_tensor
