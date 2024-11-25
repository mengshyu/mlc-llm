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
        #hidden_states = op.wrap_nested(
        #    relax.BlockBuilder()
        #    .current()
        #    .match_cast(
        #        hidden_states._expr,
        #        relax.TensorStructInfo(
        #            [hidden_states.shape[0], 3072], hidden_states.dtype
        #        ),
        #    ),
        #    "hidden_states",
        #)
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

    def dyn_repeat_4d_tensor(self, input, r0, r1, r2, r3) -> Tensor:
        assert 4 == input.ndim, "input should be 4D data tensor"
        def create_dyn_repeat_func(dtype):
            @T.prim_func
            def dyn_repeat_4d_tensor_func(
                    input: T.handle,
                    output: T.handle,
                    ch0: T.int64(),
                    ch1: T.int64(),
                    ch2: T.int64(),
                    ch3: T.int64()
                    ):
                T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
                n, c, h, w = T.int64(), T.int64(), T.int64(), T.int64()
                input_buf = T.match_buffer(input, (n, c, h, w), dtype=dtype)
                out_buf = T.match_buffer(output, (n * ch0, c * ch1, h * ch2, w * ch3), dtype=dtype)

                for n_idx in T.thread_binding(n * ch0, thread="blockIdx.x"):
                    for c_idx in T.thread_binding(c * ch1, thread="blockIdx.y"):
                        for h_idx, w_idx in T.grid(h * ch2, w * ch3):
                            with T.block("dyn_repeat_4d_tensor"):
                                T.reads(input_buf[n_idx, c_idx, h_idx, w_idx])
                                T.writes(out_buf[n_idx, c_idx, h_idx, w_idx])
                                out_buf[n_idx, c_idx, h_idx, w_idx] = input_buf[n_idx % n, c_idx % c, h_idx % h, w_idx % w]
            return dyn_repeat_4d_tensor_func
            sch = tir.Schedule(dyn_repeat_4d_tensor_func)
            self.apply_schedule(sch, sch.get_block("dyn_repeat_4d_tensor"))
            return sch.mod["main"].with_attr("tir.is_scheduled", 1)

        n, c, h, w = input.shape
        out = op.tensor_ir_op(
            create_dyn_repeat_func(input.dtype),
            "dyn_repeat_4d_tensor",
            [input, r0, r1, r2, r3],
            [Tensor.placeholder([n * r0, c * r1, h * r2, w * r3], input.dtype)]
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
                                    out_buf[n_idx, c_idx, h_idx, w_idx] = input_2_buf[n_idx, c_idx, h_idx - h1, w_idx]


            return dyn_concate_dim_2_func
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
                                    out_buf[c_idx, h_idx, w_idx] = input_2_buf[c_idx, h_idx - h1, w_idx]

            return dyn_concate_dim_1_func
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
        image_features = nn.op.reshape(image_features, ([N, H, H, C])) # N, 24, 24, 1024
        image_features = nn.op.reshape(image_features, ([N, H // 2, 2, H // 2, 2, C])) # N, 12, 2, 12, 2, 1024
        image_features = nn.op.permute_dims(image_features, axes=([0, 1, 3, 2, 4, 5])) # N, 12, 12, 2, 2, 1024
        image_features = nn.op.reshape(image_features, ([N, -1, 4 * C])) # N, 144, 4096
        image_features = nn.op.reshape(image_features, ([num_images, h_crop, w_crop, H // 2, H // 2, -1]))
        image_features = nn.op.permute_dims(image_features, axes=([0, 1, 3, 2, 4, 5]))
        image_features_hd = nn.op.reshape(image_features, ([num_images, h_crop * H // 2, w_crop * H // 2, 4 * C]))
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        temp_sub_GN = self.dyn_repeat_4d_tensor(self.sub_GN, T.int64(1), T.int64(h), T.int64(1), T.int64(1))
        image_features_hd_newline = self.dyn_concate_dim_2(image_features_hd, temp_sub_GN)
        image_features_hd_newline = nn.op.reshape(image_features_hd_newline, ([num_images, -1, hid_dim]))
        return image_features_hd_newline

    # pylint: disable=too-many-locals,too-many-locals,unused-argument
    def forward(self, pixel_values: Tensor, raw_image_h, raw_image_w) -> Tensor:
        print(f"pixel shape:{pixel_values.shape}")
        img_features = self.get_img_features(pixel_values)
        img_features = nn.op.split(img_features, indices_or_sections=[1], axis=0)

        global_image_features = img_features[0]
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        #h_crop =  raw_image_h // self.image_size
        #w_crop =  raw_image_w // self.image_size
        h_crop =  672 // self.image_size
        w_crop =  672 // self.image_size
        sub_image_features = img_features[1]
        sub_image_features_hd = self.reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop)
        sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

        global_image_features_hd = nn.op.squeeze(global_image_features_hd_newline, 0)

        combined_image = self.dyn_concate_dim_1(sub_image_features_hd_newline, self.glb_GN)
        combined_image = self.dyn_concate_dim_1(combined_image, global_image_features_hd_newline)
        print("combined image shape:", combined_image.shape)
        combined_image = nn.op.squeeze(combined_image, 0)
        print("combined image shape 2:", combined_image.shape)
        output_image = self.img_projection(combined_image)
        return output_image
