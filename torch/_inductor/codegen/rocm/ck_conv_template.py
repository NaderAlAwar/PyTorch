# mypy: allow-untyped-defs
from dataclasses import asdict, dataclass
from typing import Tuple, Optional
from torch._inductor.codegen.rocm.ck_template import CKTemplate

@dataclass
class CKConvOp:
    n_dim_spatial: int
    a_layout: str
    b_layout: str
    ds_layout: Tuple[str]
    e_layout: str
    a_element_dtype: str
    b_element_dtype: str
    acc_dtype: str
    c_shuffle_dtype: str
    ds_element_dtype: str
    e_element_dtype: str
    a_elementwise_op: str
    b_elementwise_op: str
    cde_elementwise_op: str
    conv_forward_specialization: str
    gemm_specialization: str

    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    ak1: int
    bk1: int
    m_per_xdl: int
    n_per_xdl: int
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    a_block_transfer_thread_cluster_lengths_ak0_m_ak1: Tuple[int, int, int]
    a_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    a_block_transfer_src_access_order: Tuple[int, int, int]
    a_block_transfer_src_vector_dim: int
    a_block_transfer_src_scalar_per_vector: int
    a_block_transfer_dst_scalar_per_vector_ak1: int
    a_block_lds_extra_m: bool

    b_block_transfer_thread_cluster_lengths_bk0_n_bk1: Tuple[int, int, int]
    b_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    b_block_transfer_src_access_order: Tuple[int, int, int]

    b_block_transfer_src_vector_dim: int
    b_block_transfer_src_scalar_per_vector: int
    b_block_transfer_dst_scalar_per_vector_bk1: int
    b_block_lds_extra_n: bool
    c_shuffle_block_transfer_scalar_per_vector_n_per_block: int
    block_gemm_pipeline_scheduler: str
    block_gemm_pipeline_version: str

    a_compute_dtype: Optional[str] = None
    b_compute_dtype: Optional[str] = None

    def name(self):
        # cpp alias for template instance
        return f"ck_device_grouped_convolution_fwd_multiple_abd_xdl_c_shuffle_v3_{self.key_name()}"

    def key_name(self):
        # TBD; must be unique per instance. Intended to use as dict key
        return "_".join(
            [
                "K"
                + field_name.replace("_", "").lower()
                + "V"
                + (
                    "x".join(map(str, iter(field_value)))
                    if isinstance(field_value, tuple)
                    else str(field_value).replace(":", "")
                )
                for field_name, field_value in self.dict_items()
            ]
        )

    def dict_items(self):
        return asdict(self).items()

class CKConvTemplate(CKTemplate):
    conv_template = r"""
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {
        auto conv = {{instance_type}} {};
        auto invoker = conv.MakeInvoker();

        using ck::index_t;

        constexpr index_t NumDTensor = {{NumDTensor}};
        constexpr index_t NDimSpatial = {{NDimSpatial}};

        const void* p_a;
        const void* p_b;
        const std::array<const void*, NumDTensor> p_ds;
        void* p_e;
        const std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths;
        const std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides;
        const std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths;
        const std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides;
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths;
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides;
        const std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths;
        const std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides;
        const std::array<index_t, NDimSpatial> conv_filter_strides;
        const std::array<index_t, NDimSpatial> conv_filter_dilations;
        const std::array<index_t, NDimSpatial> input_left_pads;
        const std::array<index_t, NDimSpatial> input_right_pads;
        const AElementwiseOperation a_element_op = PassThrough;
        const BElementwiseOperation b_element_op = PassThrough;
        const CDEElementwiseOperation cde_element_op = PassThrough;

        auto argument = conv.MakeArgument(
            p_a,
            p_b,
            p_ds,
            p_e,
            a_g_n_c_wis_lengths,
            a_g_n_c_wis_strides,
            b_g_k_c_xs_lengths,
            b_g_k_c_xs_strides,
            ds_g_n_k_wos_lengths,
            ds_g_n_k_wos_strides,
            e_g_n_k_wos_lengths,
            e_g_n_k_wos_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            a_element_op,
            b_element_op,
            cde_element_op
        );
        if (!conv.IsSupportedArgument(argument)) {
            // we do our best to statically avoid this case in `filter_op`
            std::cerr << "invalid argument for conv instance " << conv.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        if (workspace_size) {
            *workspace_size = conv.GetWorkSpaceSize(&argument);
            return 0;
        }
        // run the kernel
        float elapsed_time = invoker.Run(argument, StreamConfig{stream, /* time kernel */ false, /* log level */ kDEBUG_LOG});
        return 0;
    } // kernel definition
    } // extern C
"""

    @staticmethod
    def add_ck_conv_choices(
        choices,
        layout,
        input_nodes,
    ):
        template = CKConvTemplate(
            input_nodes,
            layout,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )

    def __init__(
        self,
        input_nodes,
        layout,
    ):
        super().__init__(
            "ck_conv_template",
            input_nodes,
            layout,
        )

    def gen_ops(self):
        return []
