from mlc_llm.model.vision import CLIPVisionConfig, CLIPVisionModel

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, Module, op
from tvm.script import tir as T
from mlc_llm.support.config import ConfigBase

class ImageProjection(Module):
    def __init__(self, config:ConfigBase):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * 4, config.hidden_size, bias=True
        )
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.hidden_size, config.hidden_size, bias=True
        )

    def forward(self, image_features:Tensor) -> Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# pylint: disable=invalid-name,missing-docstring
# pylint: disable=too-many-arguments, too-many-locals, redefined-argument-from-local
def create_get_positions_func(
    input_shape: Tensor,
    dtype:str
):
    @T.prim_func
    def get_positions_func(
        _input_ids: T.handle,
        _output: T.handle
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()

        input_ids_buf = T.match_buffer(_input_ids, (batch_size, seq_len), dtype="int32")
        out_buf = T.match_buffer(_output, (seq_len, 2), dtype=dtype)

        positions = T.alloc_buffer((input_shape[1], 2,), "int", scope="shared")
        MAX_INPUT_ID = -1 * T.int32(1e9)

        for b in T.thread_binding(input_shape[0], thread="blockIdx.x"):
            for s in T.thread_binding(input_shape[1], thread="threadIdx.x"):
                with T.block("compute_positions"):
                    vb, vs = T.axis.remap("SS", [b,s])
                    if (input_ids_buf[vb, vs] < 0) & (input_ids_buf[vb, vs] > MAX_INPUT_ID):
                        positions[vs,0] = vb
                        positions[vs,1] = vs
                    else:
                        positions[vs,0] = -1
                        positions[vs,1] = -1
    return get_positions_func


def get_positions(input_ids: Tensor):
    def _te_get_positions(input_ids: te.Tensor):
        MAX_INPUT_ID = -1 * T.int32(1e9)
        return te.compute(
                (input_ids.shape, 2),
                lambda b,s,i: te.if_then_else((input_ids[b,s] < 0) & (input_ids[b,s] > -MAX_INPUT_ID), s, -1),
                name = "te_positions"
        )
    assert 1 == input_ids.shape[0]
    return op.tensor_expr_op(_te_get_positions, "te_get_positions", [input_ids])


class Phi3ImageEmbedding(Module):
    def __init__(self, config: ConfigBase):
        super().__init__()
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.img_processor = CLIPVisionModel(config.vision_config)
        self.image_dim_out = 1024
        self.num_img_tokens = 144 #self.num_img_tokens = config.img_processor['num_img_tokens']
        self.use_hd_transform = True
        self.with_learnable_separator = True
        self.hd_transform_order = "sub_glb"

        self.glb_GN = nn.Parameter((1, 1, self.image_dim_out * 4))
        self.sub_GN = nn.Parameter((1, 1, 1, self.image_dim_out * 4))

        self.img_projection = ImageProjection(config)


    def get_img_features(self, img_embeds:Tensor) -> Tensor:
        LAYER_IDX = -2
        img_processor_output = self.img_processor(img_embeds)
        patch_feature = nn.op.split(img_processor_output, indices_or_sections=[1], axis = 1)
        return patch_feature[1]

    def find_positions(self, input_ids):
        #TODO implementation
        return input_ids
        MAX_INPUT_ID = int(1e9)
        positions = []
        input_ids_list = input_ids.data().numpy().tolist()
        for i in range(len(input_ids_list)):
            for j in range(len(input_ids_list[i])):
                if input_ids_list[i][j] < 0 and input_ids_list[i][j] > -MAX_INPUT_ID:
                    positions.append([i, j])
        return positions

    def forward(self, input_ids: Tensor, img_embeds: Tensor, img_sizes = None) -> Tensor:
        bs = 1
        image_dim_out = 1024
        image_h = 1008
        image_w = 1344
        base_feat_height = base_feat_width = 24
        input_shape = input_ids.shape
        H = base_feat_height
        h = image_h // 336
        w = image_w // 336
        B_ = h * w
        C = image_dim_out
        positions = op.tensor_ir_op(
            create_get_positions_func(
                input_shape,
                input_ids.dtype
                ),
            "get_positions_func",
            [input_ids],
            [Tensor.placeholder([input_shape[1], 2], input_ids.dtype)],
        )

        img_features = self.get_img_features(img_embeds)
        img_features = nn.op.split(img_features, indices_or_sections=[1], axis = 0)
        global_img_feature = img_features[0]
        sub_img = img_features[1]
        global_img_feature = nn.op.reshape(global_img_feature, ([1, H, H, C]))
        global_img_feature = nn.op.reshape(global_img_feature, ([1, H//2, 2, H//2, 2, C]))
        glb_img = nn.op.reshape(global_img_feature, ([1,H//2,H//2,4*C]))
        temp_glb_GN = nn.op.repeat(self.sub_GN,H//2,1)
        glb_img = nn.op.concat([glb_img, temp_glb_GN], dim=2)
        glb_img = nn.op.reshape(glb_img, ([1, -1, 4*C]))

        sub_img = nn.op.split(sub_img, indices_or_sections=[4], axis = 0)
        sub_img = sub_img[1]
        sub_img = nn.op.reshape(sub_img, ([B_,H,H,C]))
        sub_img = nn.op.reshape(sub_img, ([B_,H//2,2,H//2,2,C]))
        sub_img = nn.op.permute_dims(sub_img, axes=([0,1,3,2,4,5]))
        sub_img = nn.op.reshape(sub_img, ([B_,H//2,2,H//2,2,C]))
        sub_img = nn.op.reshape(sub_img, ([B_,-1,4*C]))
        sub_img = nn.op.reshape(sub_img, ([1, h, w, 12, 12, -1]))
        sub_img = nn.op.permute_dims(sub_img, axes=([0,1,3,2,4,5]))
        sub_img = nn.op.reshape(sub_img, ([1,h*12,w*12,4*C]))

        temp_sub_GN = nn.op.repeat(self.sub_GN,h*12,1)
        sub_img = nn.op.concat([sub_img, temp_sub_GN], dim=2)
        sub_img = nn.op.reshape(sub_img, ([1, -1, 4*C]))

        output_img = nn.op.concat([sub_img, self.glb_GN, glb_img], dim=1)

        num_img_tokens = int((h*w+1)*144 + 1 + (h+1)*12)
        img_set_tensor = self.img_projection(output_img)
        hidden_states = self.wte(input_ids)

        return hidden_states
